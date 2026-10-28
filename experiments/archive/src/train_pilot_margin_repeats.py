"""Frozen 500-class pilot: augmentation, fixed margin, teacher-guided margin.

Seed 42 reuses the completed augmentation/teacher-margin runs. Seeds 43 and
44 each retrain a clean teacher plus the three student methods. All models
use epoch 10; the validation writers, classes and corruptions never change.
"""

import argparse
import csv
import gc
import importlib.util
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

ROOT = Path("/workspace/shouxieti")
METHODS = ("aug_ce", "margin_fixed", "margin_hard")
PROTOCOL = {
    "version": 1, "training_seeds": [42, 43, 44],
    "split_and_class_selection": "unchanged original seed-42 split and 500 classes",
    "fixed_margin": 0.2, "pool": 10, "selected": 3,
    "scores": "cosine features versus classifier weights; bias excluded only in auxiliary loss",
    "fixed_loss": "mean over top-3 corrupted hard negatives of relu(0.2 - (s_y-s_j)); mean over ALL images",
    "fixed_gate": "none; no teacher loaded or used for fixed-margin training",
    "teacher_margin": "unchanged saved margin_hard implementation; teacher correctness gate and clipped teacher targets",
    "comparison_scope": "whole teacher-guided component versus teacher-free fixed margin; does not isolate gate from target value",
    "auxiliary_weight": "epoch 1: 0; epoch 2: 0.5; epochs 3-10: 1",
    "teacher": "same-seed clean_pair model, trained from scratch for 43/44; extra teacher training/inference recorded",
    "primary": "mean of frozen eight unseen validation corruptions",
    "secondary": "clean, mean of six seen corruptions, all individual conditions",
    "selection": "fixed epoch 10; no best-epoch selection or automatic tuning",
    "interpretation": "exploratory validation, not a test-set estimate or confirmation of novelty",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def require(value, message):
    if not value:
        raise RuntimeError(message)


def fixed_margin_loss(scores, labels):
    # Match the teacher-margin hard-negative selection exactly, including its pool.
    with torch.no_grad():
        ranking = scores.detach().clone()
        ranking.scatter_(1, labels[:, None], float("-inf"))
        pool = ranking.topk(min(10, scores.shape[1] - 1), dim=1).indices
        positions = scores.detach().gather(1, pool).topk(min(3, pool.shape[1]), dim=1).indices
        rivals = pool.gather(1, positions)
    gap = scores.gather(1, labels[:, None]) - scores.gather(1, rivals)
    violation = F.relu(0.2 - gap)
    return violation.mean(), (violation.detach() > 0).any(1).float().mean()


def self_checks(candidate):
    labels = torch.tensor([0], device="cuda")
    scores = torch.tensor([[0.1, 0.2, 0.0, -0.2]], device="cuda", requires_grad=True)
    loss, active = fixed_margin_loss(scores, labels)
    torch.testing.assert_close(loss, torch.tensor(0.4 / 3, device="cuda"), atol=1e-6, rtol=0)
    loss.backward()
    torch.testing.assert_close(scores.grad, torch.tensor([[-2/3, 1/3, 1/3, 0]], device="cuda"), atol=1e-6, rtol=0)
    require(active.item() == 1, "Fixed-margin active-mask check failed")
    teacher = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda")
    comparable, _ = candidate.margin_loss(scores.detach(), scores.detach(), teacher,
                                          torch.tensor([True], device="cuda"), labels, False)
    torch.testing.assert_close(loss.detach(), comparable, atol=1e-6, rtol=0)
    inactive, _ = fixed_margin_loss(torch.tensor([[1.0, 0.0, -0.1, -0.2]], device="cuda"), labels)
    require(inactive.item() == 0, "Satisfied margin should have zero loss")
    logging.info("SELF CHECKS PASS: fixed loss, gradients, inactive case, equivalence to capped teacher targets.")


def warmup_reference(directory):
    row = read_csv(directory / "metrics.csv")[0]
    return {"raw": float(row["val_raw_accuracy"]), "calibrated": float(row["val_calibrated_accuracy"])}


def prediction_metrics(path, targets, names, class_ids):
    with np.load(path, allow_pickle=False) as data:
        require(np.array_equal(data["targets"], targets), f"Validation order mismatch: {path}")
        require(data["conditions"].tolist() == names, f"Condition mismatch: {path}")
        require(np.array_equal(data["global_class_ids"], class_ids), f"Class mapping mismatch: {path}")
        predictions = data["predictions"]
        require(predictions.shape == (len(names), len(targets)), "Prediction shape mismatch")
        return {name: 100 * float((predictions[i] == targets).mean()) for i, name in enumerate(names)}


def evaluate_final(model, directory, rows, args, data_config, helper, controls, robust, transfer):
    loader = controls.make_loader(helper, rows["val"], 128, args.workers, False, args.seed + 1)
    targets = np.asarray([r[4] for r in rows["val"]], dtype=np.int64)
    metrics = {}
    for suite, adapter in (("seen", robust), ("unseen", transfer.ProtocolAdapter(robust.corrupt))):
        out = directory / suite
        out.mkdir()
        values = controls.evaluate_conditions({"calibrated_bn": model}, loader, targets, adapter, out, data_config)
        metrics[suite] = {r["condition"]: r["accuracy_pct"] for r in values}
    return metrics


def train_fixed(parent, args, rows, data_config, shared, recipe, initial_hash, reference,
                helper, controls, robust, transfer, candidate):
    out = parent / "margin_fixed"
    out.mkdir()
    controls.seed_everything(args.seed, args.threads)
    model = helper.make_model()
    require(controls.state_fingerprint(model) == initial_hash, "Fixed-margin initialization differs from baseline")
    loader = controls.make_loader(helper, rows["train"], 128, args.workers, True, args.seed)
    val = controls.make_loader(helper, rows["val"], 128, args.workers, False, args.seed + 1)
    calibration = shared["bn_calibration"]
    cal_loader = controls.make_loader(helper, [rows["train"][i] for i in calibration["indices"]],
                                      128, args.workers, False, args.seed + 2000)
    generator = torch.Generator().manual_seed(args.seed + 3000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)
    config = {**recipe, "method": "margin_fixed", "initial_state_sha256": initial_hash,
              "teacher_checkpoint": None, "teacher_checkpoint_sha256": None}
    helper.write_json(out / "config.json", config)
    history = []
    counts_total = np.zeros(len(robust.CONDITIONS), dtype=np.int64)
    logging.info("START seed=%d margin_fixed: no teacher; margin=0.2; 10 epochs", args.seed)
    for epoch in range(1, 11):
        started = time.perf_counter()
        model.train()
        n_total = skipped = clean_correct = corrupt_correct = 0
        ce_sum = aux_sum = active_sum = 0.0
        coefficient = candidate.auxiliary_weight(epoch)
        lr = optimizer.param_groups[0]["lr"]
        for step, (images, labels) in enumerate(loader, 1):
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            second, counts = controls.second_view(images, generator, robust)
            counts_total += counts
            n = len(labels)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=not args.no_amp):
                logits, features = candidate.forward_features(model, torch.cat((images, second)))
                ce = F.cross_entropy(logits, torch.cat((labels, labels)))
            auxiliary, active = fixed_margin_loss(candidate.cosine_scores(model, features[n:]), labels)
            loss = ce if coefficient == 0 else ce + coefficient * auxiliary
            require(bool(torch.isfinite(loss)) and bool(torch.isfinite(auxiliary)), "Non-finite loss")
            old_scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            skipped += int(scaler.get_scale() < old_scale)
            n_total += n
            ce_sum += ce.item() * n
            aux_sum += auxiliary.item() * n
            active_sum += active.item() * n
            clean_correct += (logits[:n].argmax(1) == labels).sum().item()
            corrupt_correct += (logits[n:].argmax(1) == labels).sum().item()
            if step == 1 or step % 200 == 0:
                logging.info("seed=%d fixed epoch=%d/10 step=%d/%d CE=%.4f aux=%.4f weight=%.2f",
                             args.seed, epoch, step, len(loader), ce_sum / n_total, aux_sum / n_total, coefficient)
        train_seconds = time.perf_counter() - started
        raw_loss, raw_acc = helper.evaluate(model, val, torch.nn.CrossEntropyLoss(), False)
        calibrated = controls.calibrate_copy(model, cal_loader)
        cal_loss, cal_acc = helper.evaluate(calibrated, val, torch.nn.CrossEntropyLoss(), False)
        history.append({"epoch": epoch, "lr": lr, "auxiliary_weight": coefficient,
                        "train_ce": ce_sum / n_total, "train_aux": aux_sum / n_total,
                        "train_active": active_sum / n_total, "train_clean_accuracy": clean_correct / n_total,
                        "train_corrupt_accuracy": corrupt_correct / n_total,
                        "val_raw_loss": raw_loss, "val_raw_accuracy": raw_acc,
                        "val_calibrated_loss": cal_loss, "val_calibrated_accuracy": cal_acc,
                        "train_source_images": n_total, "model_input_images": 2 * n_total,
                        "teacher_input_images": 0, "attempted_steps": len(loader), "skipped_amp_steps": skipped,
                        "train_seconds": train_seconds, "total_seconds": time.perf_counter() - started})
        controls.write_csv(out / "metrics.csv", history)
        logging.info("EPOCH seed=%d fixed %d: raw=%.2f%% calibrated=%.2f%% seconds=%.1f skipped=%d",
                     args.seed, epoch, 100 * raw_acc, 100 * cal_acc, history[-1]["total_seconds"], skipped)
        if epoch == 1:
            matches = abs(raw_acc - reference["raw"]) < 1e-6 and abs(cal_acc - reference["calibrated"]) < 1e-6
            helper.write_json(out / "warmup_check.json", {"matches_aug_ce_epoch1": matches,
                              "raw_accuracy": raw_acc, "calibrated_accuracy": cal_acc, "reference": reference})
            require(matches, "CE-only epoch 1 did not reproduce the same-seed augmentation baseline")
        scheduler.step()
        helper.save_checkpoint(out / "last.pt", {"model": model.state_dict(), "epoch": epoch,
                               "config": config, "val_accuracy": raw_acc})
        if epoch != 10:
            del calibrated
    helper.save_checkpoint(out / "epoch10_bn_calibrated.pt", {"model": calibrated.state_dict(), "epoch": 10,
                           "config": config, "val_accuracy": cal_acc, "bn_calibration": calibration})
    metrics = evaluate_final(calibrated, out, rows, args, data_config, helper, controls, robust, transfer)
    helper.write_json(out / "result.json", {"completed": True, "method": "margin_fixed", "metrics": metrics,
                      "test_evaluated": False, "confusion_evaluated": False, "teacher_input_images": 0,
                      "student_input_images": sum(r["model_input_images"] for r in history),
                      "skipped_amp_steps": sum(r["skipped_amp_steps"] for r in history),
                      "second_view_counts": {c["name"]: int(counts_total[i]) for i, c in enumerate(robust.CONDITIONS)}})
    return metrics


def load_teacher(path, helper, shared):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    require(checkpoint["epoch"] == 10 and checkpoint["bn_calibration"] == shared["bn_calibration"],
            "Teacher checkpoint or calibration mismatch")
    require(checkpoint["config"] == read_json(path.parent / "config.json"), "Teacher metadata mismatch")
    teacher = helper.make_model()
    teacher.load_state_dict(checkpoint["model"], strict=True)
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


def write_tables(parent, results, controls, robust, transfer):
    per_seed, summaries, paired = [], [], []
    for suite, conditions in (("seen", robust.CONDITIONS), ("unseen", transfer.CONDITIONS)):
        names = [c["name"] for c in conditions]
        for seed, methods in results.items():
            for method, data in methods.items():
                for name in names:
                    per_seed.append({"seed": seed, "method": method, "suite": suite,
                                     "condition": name, "accuracy_pct": data[suite][name]})
                per_seed.append({"seed": seed, "method": method, "suite": suite,
                                 "condition": f"mean_of_{len(names)-1}_{suite}",
                                 "accuracy_pct": float(np.mean([data[suite][n] for n in names[1:]]))})
    for suite, endpoint in (("seen", "clean"), ("seen", "mean_of_6_seen"),
                            ("unseen", "mean_of_8_unseen"), ("unseen", "lowres_20")):
        by_method = {}
        for method in METHODS:
            values = {r["seed"]: r["accuracy_pct"] for r in per_seed
                      if r["suite"] == suite and r["condition"] == endpoint and r["method"] == method}
            by_method[method] = values
            if values:
                summaries.append({"endpoint": endpoint, "method": method, "n_seeds": len(values),
                                  "seeds": ";".join(map(str, sorted(values))),
                                  "mean_accuracy_pct": float(np.mean(list(values.values()))),
                                  "sample_sd_pp": float(np.std(list(values.values()), ddof=1)) if len(values) > 1 else ""})
        for a, b in (("aug_ce", "margin_fixed"), ("aug_ce", "margin_hard"), ("margin_fixed", "margin_hard")):
            common = sorted(set(by_method[a]) & set(by_method[b]))
            if common:
                differences = [by_method[b][s] - by_method[a][s] for s in common]
                paired.append({"endpoint": endpoint, "a": a, "b": b, "n_paired_seeds": len(common),
                               "seeds": ";".join(map(str, common)), "mean_delta_pp": float(np.mean(differences)),
                               "sample_sd_delta_pp": float(np.std(differences, ddof=1)) if len(common) > 1 else "",
                               "per_seed_delta_pp": ";".join(f"{s}:{d:+.6f}" for s, d in zip(common, differences))})
    controls.write_csv(parent / "per_seed.csv", per_seed)
    controls.write_csv(parent / "summary.csv", summaries)
    if paired:
        controls.write_csv(parent / "paired_deltas.csv", paired)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-run", type=Path, required=True)
    opts = parser.parse_args()
    old = opts.candidates_run.resolve()
    require(old.is_relative_to(ROOT.resolve()), "Candidate directory must be inside the project")
    original = read_json(old / "config.json")
    run, transfer_run = Path(original["controls_run"]).resolve(), Path(original["transfer_run"]).resolve()
    require(all(p.is_relative_to(ROOT.resolve()) for p in (run, transfer_run)), "Reference directory outside project")
    require(read_json(old / "result.json")["completed"], "Original candidates incomplete")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Expose only allocated GPU 0")
    shared = read_json(run / "config.json")
    transfer_config = read_json(transfer_run / "config.json")
    require(transfer_config["controls_config"] == shared, "Reference configs differ")
    require(str(torch.__version__) == shared["torch"] and str(torchvision.__version__) == shared["torchvision"],
            "Use the original torch/torchvision versions")
    sources = [("helper", run / "original_train_pilot.py", shared["helper_sha256"]),
               ("controls", run / "train_pilot_controls.py", shared["source_sha256"]),
               ("robust", run / "eval_pilot_robustness.py", shared["robustness_source_sha256"]),
               ("transfer", transfer_run / "eval_pilot_transfer.py", transfer_config["source_sha256"]),
               ("candidate", old / "train_pilot_candidates.py", original["source_sha256"])]
    import hashlib
    for name, path, expected in sources:
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"Changed source snapshot: {name}")
    modules = {name: load_module(f"repeat_{name}", path) for name, path, _ in sources}
    helper, controls, robust, transfer, candidate = [modules[n] for n in ("helper", "controls", "robust", "transfer", "candidate")]
    rows, data_config, hashes = helper.load_data(ROOT)
    require(hashes == shared["data_sha256"] == original["data_sha256"], "Data or split changed")
    for path, expected in (("pilot_robustness_v1.json", shared["evaluation_protocol"]),
                           ("pilot_transfer_v1.json", transfer_config["protocol"]),
                           ("bn_calibration_pilot_v1.json", shared["bn_calibration"])):
        require(read_json(ROOT / "configs" / path) == expected, f"Changed protocol: {path}")
    require(helper.sha256(old / "val_sample_index.csv") == helper.sha256(run / "val_sample_index.csv")
            == helper.sha256(transfer_run / "val_sample_index.csv"), "Sample indexes differ")
    base_args = argparse.Namespace(**shared["args"])
    require(base_args.seed == 42 and base_args.epochs == 10 and base_args.batch_size == 128, "Unexpected original recipe")
    parent = ROOT / "outputs" / f"margin_repeats_{datetime.now():%Y%m%d_%H%M%S_%f}"
    parent.mkdir(parents=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.StreamHandler(),
        logging.FileHandler(ROOT / "logs" / f"{parent.name}.log", encoding="utf-8")])
    logging.info("Output: %s; GPU=%s; 9 new training jobs, each 10 epochs", parent, torch.cuda.get_device_name(0))
    controls.seed_everything(42, base_args.threads)
    self_checks(candidate)
    robust.fixed_json(ROOT / "configs/pilot_margin_repeats_v1.json", PROTOCOL)
    recipe = {"protocol": PROTOCOL, "args": vars(base_args), "data_config": data_config, "data_sha256": hashes,
              "original_candidates": str(old), "controls_run": str(run), "transfer_run": str(transfer_run),
              "source_sha256": helper.sha256(Path(__file__)),
              "source_snapshots": {name: {"path": str(path), "sha256": expected} for name, path, expected in sources},
              "bn_calibration": shared["bn_calibration"], "seen_protocol": shared["evaluation_protocol"],
              "unseen_protocol": transfer_config["protocol"], "torch": str(torch.__version__),
              "torchvision": str(torchvision.__version__), "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0), "test_evaluated": False, "confusion_evaluated": False}
    helper.write_json(parent / "config.json", recipe)
    shutil.copyfile(Path(__file__), parent / Path(__file__).name)
    for _, path, _ in sources:
        shutil.copyfile(path, parent / path.name)
    shutil.copyfile(run / "val_sample_index.csv", parent / "val_sample_index.csv")
    targets = np.asarray([r[4] for r in rows["val"]], dtype=np.int64)
    class_ids = data_config["pilot_classes.json"]["global_class_ids"]
    results = {42: {"aug_ce": {}, "margin_hard": {}}}
    for suite, conditions in (("seen", robust.CONDITIONS), ("unseen", transfer.CONDITIONS)):
        names = [c["name"] for c in conditions]
        baseline = run / "aug_ce/predictions_calibrated_bn.npz" if suite == "seen" else transfer_run / "predictions_aug_ce.npz"
        results[42]["aug_ce"][suite] = prediction_metrics(baseline, targets, names, class_ids)
        results[42]["margin_hard"][suite] = prediction_metrics(old / "margin_hard" / suite / "predictions_calibrated_bn.npz", targets, names, class_ids)
    helper.write_json(parent / "reused_seed42.json", {"aug_ce": str(run / "aug_ce"), "margin_hard": str(old / "margin_hard"),
                      "teacher": str(run / "clean_pair"), "selection": "existing calibrated epoch 10"})
    fixed_parent = parent / "seed42"
    fixed_parent.mkdir()
    results[42]["margin_fixed"] = train_fixed(fixed_parent, base_args, rows, data_config, shared, recipe,
        read_json(run / "initial_state.json")["sha256"], warmup_reference(run / "aug_ce"),
        helper, controls, robust, transfer, candidate)
    write_tables(parent, results, controls, robust, transfer)
    for seed in (43, 44):
        args = argparse.Namespace(**{**vars(base_args), "seed": seed})
        seed_dir = parent / f"seed{seed}"
        control_dir, student_dir = seed_dir / "controls", seed_dir / "students"
        control_dir.mkdir(parents=True)
        student_dir.mkdir()
        seed_shared = {**shared, "args": vars(args), "augmentation_seed": seed + 3000,
                       "repeat_source": str(parent), "source_sha256": shared["source_sha256"]}
        helper.write_json(control_dir / "config.json", seed_shared)
        results[seed] = {}
        for method in ("clean_pair", "aug_ce"):
            controls.train_one(method, control_dir, args, rows, data_config, shared["bn_calibration"],
                               shared["evaluation_protocol"], helper, robust, seed_shared)
            gc.collect()
            torch.cuda.empty_cache()
        # Load the augmentation checkpoint only for the same frozen unseen suite.
        baseline = load_teacher(control_dir / "aug_ce/epoch10_bn_calibrated.pt", helper, shared)
        results[seed]["aug_ce"] = evaluate_final(baseline, control_dir / "aug_ce", rows, args, data_config,
                                                 helper, controls, robust, transfer)
        del baseline
        gc.collect()
        torch.cuda.empty_cache()
        initial_hash = read_json(control_dir / "initial_state.json")["sha256"]
        reference = warmup_reference(control_dir / "aug_ce")
        seed_recipe = {**recipe, "args": vars(args), "controls_run": str(control_dir)}
        results[seed]["margin_fixed"] = train_fixed(student_dir, args, rows, data_config, shared, seed_recipe,
            initial_hash, reference, helper, controls, robust, transfer, candidate)
        write_tables(parent, results, controls, robust, transfer)
        gc.collect()
        torch.cuda.empty_cache()
        teacher_path = control_dir / "clean_pair/epoch10_bn_calibrated.pt"
        teacher = load_teacher(teacher_path, helper, shared)
        candidate.self_checks(teacher)
        hard_recipe = {**seed_recipe, "teacher_checkpoint": str(teacher_path),
                       "teacher_checkpoint_sha256": helper.sha256(teacher_path), "loss_protocol": candidate.LOSS_PROTOCOL}
        results[seed]["margin_hard"] = candidate.train_one("margin_hard", student_dir, args, rows, data_config,
            teacher, helper, controls, robust, transfer, shared, hard_recipe, initial_hash, reference)
        del teacher
        write_tables(parent, results, controls, robust, transfer)
        gc.collect()
        torch.cuda.empty_cache()
    helper.write_json(parent / "result.json", {"completed": True, "training_seeds": [42, 43, 44],
                      "new_training_jobs": 9, "extra_clean_teacher_jobs": 2, "results": results,
                      "test_evaluated": False, "confusion_evaluated": False,
                      "interpretation": "mean and sample SD over training seeds on the SAME validation data; not a confidence interval"})
    logging.info("DONE: %s", parent)
    print((parent / "summary.csv").read_text(encoding="utf-8"), flush=True)
    print((parent / "paired_deltas.csv").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
