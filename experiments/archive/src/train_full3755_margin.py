"""HWDB1.1 full 3755-class scale check: paired augmentation vs fixed margin.

Reuses the exact training functions frozen by the completed pilot repeats.
Only the head/class coverage and training-only BN calibration sample change.
Ten epochs, seed 42, fixed final checkpoint; no official test image is read.
"""

import argparse
import csv
import gc
import importlib.util
import json
import logging
import os
import random
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torchvision

ROOT = Path("/workspace/shouxieti")


def require(value, message):
    if not value:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_model():
    # Same construction order as the pilot; only the output head size changes.
    model = torchvision.models.resnet18(weights=None)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    torch.nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    model.maxpool = torch.nn.Identity()
    model.fc = torch.nn.Linear(model.fc.in_features, 3755)
    return model.cuda()


def load_full_data(root, helper, expected_classes=3755):
    files = ("classes.json", "pilot_classes.json", "writer_splits.json", "preprocess.json")
    configs = {name: read_json(root / "configs" / name) for name in files}
    classes = configs["classes.json"]["classes"]
    require(len(classes) == expected_classes and len(set(classes)) == expected_classes, "Invalid full class mapping")
    require(configs["preprocess.json"] == helper.EXPECTED_PREPROCESS, "Preprocessing changed")
    splits = configs["writer_splits.json"]["splits"]
    counts = {"train": 200, "confusion": 20, "val": 20, "test": 60}
    for name, count in counts.items():
        require(len(splits[name]) == count and len(set(splits[name])) == count, f"Invalid writer split: {name}")
    require(len(set(w for name in counts for w in splits[name])) == 300, "Writer overlap")
    writer_sets = {name: set(values) for name, values in splits.items()}
    rows = {"train": [], "val": []}
    observed = {name: set() for name in rows}
    class_counts = {name: Counter() for name in rows}
    paths, file_sizes, last_end = {}, {}, {}
    index = root / "data/processed/hwdb1_1_index.csv"
    with index.open(encoding="utf-8", newline="") as stream:
        for record in csv.DictReader(stream):
            split = record["split"]
            require(split in writer_sets, "Unknown split in index")
            writer, label = int(record["writer_id"]), int(record["class_id"])
            require(writer in writer_sets[split], "Index writer/split mismatch")
            require(0 <= label < expected_classes, "Invalid global class ID")
            if split not in rows:
                continue  # Do not open test/confusion images or GNT files.
            relative = record["gnt_path"]
            require(int(Path(relative).stem.split('-')[0]) == writer, "GNT filename writer mismatch")
            if relative not in paths:
                path = (root / relative).resolve()
                require(path.is_relative_to(root.resolve()) and path.is_file(), f"Invalid train/val path: {relative}")
                paths[relative] = str(path)
                file_sizes[relative] = path.stat().st_size
            offset, width, height = [int(record[k]) for k in ("pixel_offset", "width", "height")]
            require(offset >= 10 and width > 0 and height > 0, "Invalid GNT dimensions or offset")
            require(offset + width * height <= file_sizes[relative], "Image extends beyond GNT file")
            require(offset >= last_end.get(relative, 0), "Repeated, overlapping or out-of-order indexed image")
            last_end[relative] = offset + width * height
            rows[split].append((paths[relative], offset, width, height, label))
            observed[split].add(writer)
            class_counts[split][label] += 1
    audit = read_json(root / "outputs/data_audit.json")
    for split in rows:
        require(observed[split] == writer_sets[split], f"Missing writers in {split}")
        require(set(class_counts[split]) == set(range(expected_classes)), f"Missing classes in {split}")
        require(len(rows[split]) == audit["samples_per_split"][split], f"Sample count differs from audit: {split}")
    hashes = {name: helper.sha256(root / "configs" / name) for name in files}
    hashes["hwdb1_1_index.csv"] = helper.sha256(index)
    return rows, configs, hashes, class_counts


def full_protocol(args, hashes, repeats):
    return {"version": 1, "stage": "full-class scale check, not final test evaluation", "classes": 3755,
            "training_seed": args.seed, "epochs": 10, "batch_size": 128,
            "training_writers": 200, "validation_writers": 20,
            "methods": ["aug_ce", "margin_fixed"], "margin": repeats.PROTOCOL["fixed_margin"],
            "pool": repeats.PROTOCOL["pool"], "selected": repeats.PROTOCOL["selected"],
            "loss": repeats.PROTOCOL["fixed_loss"], "auxiliary_weight": repeats.PROTOCOL["auxiliary_weight"],
            "training": "same source batches, paired clean/second views, initialization and LR schedule for both methods",
            "checkpoint_selection": "fixed epoch 10; no validation-max selection",
            "initialization": "random; no pilot checkpoint or teacher",
            "primary": "mean of the same eight training-unseen synthetic validation conditions",
            "secondary": "clean, six seen-condition mean, all individual conditions",
            "data_sha256": hashes, "test_evaluated": False, "confusion_evaluated": False,
            "boundary": "200-writer subset of HWDB1.1 training split; not the HWDB1.0+1.1 / ICDAR2013 benchmark protocol",
            "convergence": "10 epochs is an initial matched budget; inspect learning curves before claiming convergence"}


def comparison_rows(results, conditions):
    rows = []
    for condition in conditions:
        name = condition["name"]
        a, b = results["aug_ce"][name], results["margin_fixed"][name]
        rows.append({"condition": name, "aug_ce_pct": a, "margin_fixed_pct": b, "fixed_minus_aug_pp": b - a})
    a = float(np.mean([r["aug_ce_pct"] for r in rows[1:]]))
    b = float(np.mean([r["margin_fixed_pct"] for r in rows[1:]]))
    rows.append({"condition": f"mean_of_{len(conditions)-1}_corruptions", "aug_ce_pct": a,
                 "margin_fixed_pct": b, "fixed_minus_aug_pp": b - a})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats-run", type=Path, required=True)
    opts = parser.parse_args()
    run = opts.repeats_run.resolve()
    require(run.is_relative_to(ROOT.resolve()), "Reference run must be inside /workspace/shouxieti")
    prior = read_json(run / "config.json")
    require(read_json(run / "result.json")["completed"], "Pilot repeats did not complete")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Expose exactly the allocated GPU 0")
    require(str(torch.__version__) == prior["torch"] and str(torchvision.__version__) == prior["torchvision"],
            "Use the same torch and torchvision versions")
    import hashlib
    sources = {name: (run / Path(item["path"]).name, item["sha256"]) for name, item in prior["source_snapshots"].items()}
    sources["repeats"] = (run / "train_pilot_margin_repeats.py", prior["source_sha256"])
    for name, (path, expected) in sources.items():
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"Changed source snapshot: {name}")
    modules = {name: load_module(f"full3755_{name}", path) for name, (path, _) in sources.items()}
    original, controls, robust, transfer, candidate, repeats = [modules[name] for name in
        ("helper", "controls", "robust", "transfer", "candidate", "repeats")]
    rows, configs, hashes, class_counts = load_full_data(ROOT, original)
    require(hashes == prior["data_sha256"], "Data, split, or preprocessing changed since pilot")
    for filename, expected in (("pilot_robustness_v1.json", prior["seen_protocol"]),
                               ("pilot_transfer_v1.json", prior["unseen_protocol"]),
                               ("bn_calibration_pilot_v1.json", prior["bn_calibration"])):
        require(read_json(ROOT / "configs" / filename) == expected, f"Changed prior protocol: {filename}")
    require([c["name"] for c in robust.CONDITIONS] == [c["name"] for c in prior["seen_protocol"]["conditions"]], "Seen conditions differ")
    require([c["name"] for c in transfer.CONDITIONS] == [c["name"] for c in prior["unseen_protocol"]["conditions"]], "Unseen conditions differ")
    args = argparse.Namespace(**prior["args"])
    require(args.seed == 42 and args.epochs == 10 and args.batch_size == 128, "Expected 10-epoch seed-42 recipe")
    args.mode = "full3755"
    parent = ROOT / "outputs" / f"full3755_{datetime.now():%Y%m%d_%H%M%S_%f}_seed42"
    parent.mkdir(parents=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.StreamHandler(),
        logging.FileHandler(ROOT / "logs" / f"{parent.name}.log", encoding="utf-8")])
    logging.info("Output: %s", parent)
    logging.info("Classes=3755 train=%d val=%d; GPU=%s; steps/epoch=%d",
                 len(rows["train"]), len(rows["val"]), torch.cuda.get_device_name(0), (len(rows["train"]) + 127) // 128)
    controls.seed_everything(args.seed, args.threads)
    repeats.self_checks(candidate)
    # Adapter only changes construction. Frozen dataset/optimizer/evaluation implementations are reused.
    helper = SimpleNamespace(GNTDataset=original.GNTDataset, seed_worker=original.seed_worker,
        make_model=make_model, evaluate=original.evaluate, write_json=original.write_json,
        save_checkpoint=original.save_checkpoint, sha256=original.sha256)
    probe = make_model().eval()
    with torch.no_grad():
        logits = probe(torch.zeros(2, 1, 64, 64, device="cuda"))
        require(logits.shape == (2, 3755) and bool(torch.isfinite(logits).all()), "Full output head check failed")
    del probe, logits
    torch.cuda.empty_cache()
    protocol = full_protocol(args, hashes, repeats)
    robust.fixed_json(ROOT / "configs/full3755_margin_v1.json", protocol)
    calibration = {"version": 1, "scope": "full3755", "split": "train", "seed": 20260920, "batch_size": 128,
                   "indices": random.Random(20260920).sample(range(len(rows["train"])), 8192),
                   "data_sha256": hashes, "train_rows": len(rows["train"])}
    robust.fixed_json(ROOT / "configs/bn_calibration_full3755_v1.json", calibration)
    class_map = {"global_class_ids": list(range(3755)), "classes": configs["classes.json"]["classes"]}
    # Legacy evaluator expects this key. This is a separate explicit 3755-class view,
    # NEVER a write to the original configs/pilot_classes.json file.
    evaluation_data = {"pilot_classes.json": class_map}
    shared = {"args": vars(args), "data_config": configs, "data_sha256": hashes,
              "evaluation_class_mapping": class_map, "legacy_evaluator_mapping_key": "pilot_classes.json",
              "source_sha256": helper.sha256(Path(__file__)), "source_repeats": str(run),
              "source_snapshots": {n: {"path": str(p), "sha256": h} for n, (p, h) in sources.items()},
              "protocol": protocol, "bn_calibration": calibration,
              "seen_protocol": prior["seen_protocol"], "unseen_protocol": prior["unseen_protocol"],
              "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}
    helper.write_json(parent / "config.json", shared)
    helper.write_json(parent / "classes_full3755.json", class_map)
    helper.write_json(parent / "data_audit.json", {"classes": 3755, "samples": {s: len(v) for s, v in rows.items()},
        "class_count_range": {s: {"min": min(c.values()), "max": max(c.values())} for s, c in class_counts.items()},
        "calibration_rows": 8192, "test_evaluated": False, "confusion_evaluated": False})
    with (parent / "val_sample_index.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row", "gnt_path", "pixel_offset", "width", "height", "writer_id", "local_class_id", "global_class_id", "character"])
        for i, (path, offset, width, height, label) in enumerate(rows["val"]):
            writer.writerow([i, Path(path).relative_to(ROOT).as_posix(), offset, width, height,
                             int(Path(path).stem.split('-')[0]), label, label, class_map["classes"][label]])
    shutil.copyfile(Path(__file__), parent / Path(__file__).name)
    for path, _ in sources.values():
        shutil.copyfile(path, parent / path.name)
    logging.info("DATA/HEAD CHECKS PASS: full classes, writer separation, offsets, hashes and 3755-output model")
    results = {"aug_ce": {}}
    results["aug_ce"]["seen"] = controls.train_one("aug_ce", parent, args, rows, evaluation_data, calibration,
        prior["seen_protocol"], helper, robust, shared)
    checkpoint = torch.load(parent / "aug_ce/epoch10_bn_calibrated.pt", map_location="cpu", weights_only=False)
    baseline = make_model()
    baseline.load_state_dict(checkpoint["model"], strict=True)
    baseline.eval()
    del checkpoint
    loader = controls.make_loader(helper, rows["val"], 128, args.workers, False, args.seed + 1)
    targets = np.asarray([r[4] for r in rows["val"]], dtype=np.int64)
    unseen_dir = parent / "aug_ce/unseen"
    unseen_dir.mkdir()
    values = controls.evaluate_conditions({"calibrated_bn": baseline}, loader, targets,
        transfer.ProtocolAdapter(robust.corrupt), unseen_dir, evaluation_data)
    results["aug_ce"]["unseen"] = {r["condition"]: r["accuracy_pct"] for r in values}
    del baseline, loader
    gc.collect()
    torch.cuda.empty_cache()
    results["margin_fixed"] = repeats.train_fixed(parent, args, rows, evaluation_data, shared, shared,
        read_json(parent / "initial_state.json")["sha256"], repeats.warmup_reference(parent / "aug_ce"),
        helper, controls, robust, transfer, candidate)
    for suite, conditions in (("seen", robust.CONDITIONS), ("unseen", transfer.CONDITIONS)):
        table = comparison_rows({m: result[suite] for m, result in results.items()}, conditions)
        controls.write_csv(parent / f"comparison_{suite}.csv", table)
        print((parent / f"comparison_{suite}.csv").read_text(encoding="utf-8"), flush=True)
    helper.write_json(parent / "result.json", {"completed": True, "classes": 3755, "seed": 42,
        "results": results, "test_evaluated": False, "confusion_evaluated": False,
        "checkpoint_selection": "fixed epoch 10", "stage": "full-class scale check on held-out validation writers"})
    logging.info("DONE: %s", parent)


if __name__ == "__main__":
    main()
