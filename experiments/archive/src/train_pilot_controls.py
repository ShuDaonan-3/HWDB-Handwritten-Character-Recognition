"""Matched pilot controls: two clean views versus clean + corrupted views.

Both start from identical random initialization, use identical source batches,
and train 10 epochs with 2B images per forward and plain averaged cross-entropy.
Final evaluation uses the previously frozen validation-corruption protocol.
"""

import argparse
import copy
import csv
import gc
import hashlib
import importlib.util
import json
import logging
import os
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.swa_utils import update_bn
from torch.utils.data import DataLoader
import torchvision

ROOT = Path("/workspace/shouxieti")
METHODS = ("clean_pair", "aug_ce")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def seed_everything(seed, threads):
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def state_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str((array.dtype, array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def make_loader(helper, rows, batch_size, workers, shuffle, seed):
    return DataLoader(helper.GNTDataset(rows), batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=True, persistent_workers=(workers > 0),
                      worker_init_fn=helper.seed_worker,
                      generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def second_view(images, generator, robust):
    choices = torch.randint(1, len(robust.CONDITIONS), (images.shape[0],), generator=generator)
    result = torch.empty_like(images)
    for index, condition in enumerate(robust.CONDITIONS[1:], 1):
        selected = torch.where(choices == index)[0].to(images.device)
        if selected.numel():
            result[selected] = robust.corrupt(images.index_select(0, selected), condition)
    return result, torch.bincount(choices, minlength=len(robust.CONDITIONS)).numpy()


@torch.no_grad()
def calibrate_copy(model, loader):
    calibrated = copy.deepcopy(model)
    update_bn(loader, calibrated, device=torch.device("cuda:0"))
    calibrated.eval()
    if not all(torch.equal(a, b) for a, b in zip(model.parameters(), calibrated.parameters())):
        raise RuntimeError("BN calibration modified a learned parameter.")
    return calibrated


@torch.no_grad()
def evaluate_conditions(models, loader, targets, robust, out, data_config):
    shape = (len(robust.CONDITIONS), len(targets))
    predictions = {name: np.empty(shape, dtype=np.int64) for name in models}
    confidence = {name: np.empty(shape, dtype=np.float32) for name in models}
    losses = {name: [] for name in models}
    saved_buffers = {name: {k: v.detach().clone() for k, v in model.named_buffers()}
                     for name, model in models.items()}
    for model in models.values():
        model.eval()
    for ci, condition in enumerate(robust.CONDITIONS):
        sums = {name: 0.0 for name in models}
        offset = 0
        for images, labels in loader:
            n = labels.numel()
            if not np.array_equal(labels.numpy(), targets[offset:offset + n]):
                raise RuntimeError("Validation order changed.")
            images = robust.corrupt(images.cuda(non_blocking=True), condition)
            labels_gpu = labels.cuda(non_blocking=True)
            for name, model in models.items():
                logits = model(images)  # All final measurements use FP32.
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Non-finite validation logits.")
                score, predicted = logits.softmax(1).max(1)
                predictions[name][ci, offset:offset + n] = predicted.cpu().numpy()
                confidence[name][ci, offset:offset + n] = score.cpu().numpy()
                sums[name] += F.cross_entropy(logits, labels_gpu, reduction="sum").item()
            offset += n
        if offset != len(targets):
            raise RuntimeError("Validation size changed.")
        for name in models:
            losses[name].append(sums[name] / len(targets))
            logging.info("%s EVAL %s %s accuracy=%.2f%%", out.name, name, condition["name"],
                         100 * float((predictions[name][ci] == targets).mean()))
    summary = []
    for name, model in models.items():
        if any(not torch.equal(value, saved_buffers[name][key]) for key, value in model.named_buffers()):
            raise RuntimeError("BN buffers changed during validation.")
        for ci, row in enumerate(robust.summarize(predictions[name], targets)):
            summary.append({"bn_variant": name, **row, "cross_entropy": losses[name][ci]})
        np.savez_compressed(out / f"predictions_{name}.npz", predictions=predictions[name],
                            confidence=confidence[name], targets=targets,
                            conditions=np.asarray([c["name"] for c in robust.CONDITIONS]),
                            global_class_ids=np.asarray(data_config["pilot_classes.json"]["global_class_ids"]),
                            classes=np.asarray(data_config["pilot_classes.json"]["classes"]))
    write_csv(out / "summary.csv", summary)
    return summary


def train_one(method, parent, args, rows, data_config, calibration, protocol, helper, robust, shared_config):
    out = parent / method
    out.mkdir(exist_ok=False)
    seed_everything(args.seed, args.threads)
    model = helper.make_model()  # Fresh identical initialization for both controls.
    initial_hash = state_fingerprint(model)
    initial_path = parent / "initial_state.json"
    if initial_path.exists() and read_json(initial_path)["sha256"] != initial_hash:
        raise RuntimeError("The two controls did not receive identical initialization.")
    if not initial_path.exists():
        write_json(initial_path, {"sha256": initial_hash, "seed": args.seed})
    train_loader = make_loader(helper, rows["train"], args.batch_size, args.workers, True, args.seed)
    val_loader = make_loader(helper, rows["val"], 128, args.workers, False, args.seed + 1)
    calibration_loader = make_loader(helper, [rows["train"][i] for i in calibration["indices"]],
                                     128, args.workers, False, args.seed + 2000)
    aug_generator = torch.Generator().manual_seed(args.seed + 3000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)
    criterion = nn.CrossEntropyLoss()
    config = {**shared_config, "method": method,
              "initial_state_sha256": initial_hash,
              "training": "mean CE over concatenated clean/second-view batch; no teacher or extra loss",
              "second_view": "identical clean copy" if method == "clean_pair" else "uniform per-image draw from the six fixed corruptions",
              "raw_input_images_per_full_step": args.batch_size,
              "model_input_images_per_full_step": 2 * args.batch_size,
              "checkpoint_selection": "fixed epoch 10; epoch validation is diagnostic only"}
    write_json(out / "config.json", config)
    logging.info("START %s: epochs=%d source_batch=%d model_batch=%d steps_per_epoch=%d",
                 method, args.epochs, args.batch_size, 2 * args.batch_size, len(train_loader))
    history = []
    condition_counts = np.zeros(len(robust.CONDITIONS), dtype=np.int64)
    total_skips = 0
    final_calibrated = None
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        loss_sum = count = clean_correct = second_correct = skipped = 0
        lr = optimizer.param_groups[0]["lr"]
        for step, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            n = labels.numel()
            if method == "aug_ce":
                second, counts = second_view(images, aug_generator, robust)
                condition_counts += counts
            else:
                second = images
                condition_counts[0] += n
            paired_images = torch.cat((images, second), dim=0)
            paired_labels = torch.cat((labels, labels), dim=0)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=not args.no_amp):
                logits = model(paired_images)
                loss = criterion(logits, paired_labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss.")
            previous_scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            skipped += int(scaler.get_scale() < previous_scale)
            loss_sum += loss.item() * n
            count += n
            clean_correct += (logits[:n].argmax(1) == labels).sum().item()
            second_correct += (logits[n:].argmax(1) == labels).sum().item()
            if step == 1 or step % 200 == 0:
                logging.info("%s epoch=%d/10 step=%d/%d loss=%.4f clean_acc=%.2f%% second_acc=%.2f%%",
                             method, epoch, step, len(train_loader), loss_sum / count,
                             100 * clean_correct / count, 100 * second_correct / count)
        train_seconds = time.perf_counter() - started
        raw_val_loss, raw_val_acc = helper.evaluate(model, val_loader, criterion, False)
        calibrated = calibrate_copy(model, calibration_loader)
        calibrated_loss, calibrated_acc = helper.evaluate(calibrated, val_loader, criterion, False)
        total_skips += skipped
        history.append({"epoch": epoch, "lr": lr, "train_loss": loss_sum / count,
                        "train_clean_accuracy": clean_correct / count,
                        "train_second_accuracy": second_correct / count,
                        "val_raw_loss": raw_val_loss, "val_raw_accuracy": raw_val_acc,
                        "val_calibrated_loss": calibrated_loss, "val_calibrated_accuracy": calibrated_acc,
                        "train_source_images": count, "model_input_images": 2 * count,
                        "attempted_steps": len(train_loader), "skipped_amp_steps": skipped,
                        "train_seconds": train_seconds, "total_seconds": time.perf_counter() - started})
        write_csv(out / "metrics.csv", history)
        logging.info("EPOCH %s %d: raw_val=%.2f%% calibrated_val=%.2f%% train_seconds=%.1f skipped_steps=%d",
                     method, epoch, 100 * raw_val_acc, 100 * calibrated_acc, train_seconds, skipped)
        scheduler.step()
        helper.save_checkpoint(out / "last.pt", {"model": model.state_dict(), "epoch": epoch,
                               "config": config, "val_accuracy": raw_val_acc})
        if epoch == args.epochs:
            final_calibrated = calibrated
            helper.save_checkpoint(out / "epoch10_bn_calibrated.pt", {
                "model": calibrated.state_dict(), "epoch": epoch, "config": config,
                "val_accuracy": calibrated_acc, "bn_calibration": calibration})
        else:
            del calibrated
    targets = np.asarray([r[4] for r in rows["val"]], dtype=np.int64)
    summary = evaluate_conditions({"raw_bn": model, "calibrated_bn": final_calibrated},
                                  val_loader, targets, robust, out, data_config)
    write_json(out / "result.json", {"completed": True, "method": method, "epochs": args.epochs,
               "attempted_steps": args.epochs * len(train_loader), "skipped_amp_steps": total_skips,
               "successful_optimizer_steps": args.epochs * len(train_loader) - total_skips,
               "second_view_counts": {c["name"]: int(condition_counts[i]) for i, c in enumerate(robust.CONDITIONS)},
               "test_evaluated": False, "confusion_evaluated": False, "summary": summary})
    return {row["condition"]: row["accuracy_pct"] for row in summary if row["bn_variant"] == "calibrated_bn"}


def compare_results(reference, clean, augmented, conditions):
    comparison = []
    for condition in conditions:
        name = condition["name"]
        comparison.append({"condition": name, "original_single_view_pct": reference[name],
                           "clean_pair_pct": clean[name], "aug_ce_pct": augmented[name],
                           "aug_minus_clean_pair_pp": augmented[name] - clean[name]})
    names = [c["name"] for c in conditions[1:]]
    comparison.append({"condition": "mean_of_6_corruptions",
                       "original_single_view_pct": sum(reference[n] for n in names) / len(names),
                       "clean_pair_pct": sum(clean[n] for n in names) / len(names),
                       "aug_ce_pct": sum(augmented[n] for n in names) / len(names),
                       "aug_minus_clean_pair_pp": sum(augmented[n] - clean[n] for n in names) / len(names)})
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--robustness-run", type=Path, required=True)
    options = parser.parse_args()
    reference_run, robustness_run = options.reference_run.resolve(), options.robustness_run.resolve()
    if any(not p.is_relative_to(ROOT.resolve()) for p in (reference_run, robustness_run)):
        raise SystemExit("Both run directories must be under /workspace/shouxieti.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("Expose exactly one allocated GPU with CUDA_VISIBLE_DEVICES.")
    original = read_json(reference_run / "config.json")
    args = argparse.Namespace(**original["args"])
    if args.mode != "pilot" or args.epochs != 10 or args.batch_size != 128:
        raise SystemExit("Expected the original 10-epoch, batch-128 pilot.")
    if str(torch.__version__) != original["torch"] or str(torchvision.__version__) != original["torchvision"]:
        raise SystemExit("Keep the original torch/torchvision versions for the matched controls.")
    helper_path = reference_run / "train_pilot.py"
    helper = load_module("controls_original_pilot", helper_path)
    if helper.sha256(helper_path) != original["source_sha256"]:
        raise SystemExit("Original helper snapshot changed.")
    robustness_config = read_json(robustness_run / "config.json")
    if robustness_config["original_config"] != original:
        raise SystemExit("Robustness results do not belong to the specified pilot.")
    robust_path = robustness_run / "eval_pilot_robustness.py"
    if helper.sha256(robust_path) != robustness_config["source_sha256"]:
        raise SystemExit("Robustness source snapshot changed.")
    robust = load_module("controls_robustness_protocol", robust_path)
    rows, data_config, hashes = helper.load_data(ROOT)
    if hashes != original["data_sha256"]:
        raise SystemExit("Data fingerprints changed.")
    protocol = read_json(ROOT / "configs/pilot_robustness_v1.json")
    calibration = read_json(ROOT / "configs/bn_calibration_pilot_v1.json")
    if protocol != robustness_config["protocol"] or calibration != robustness_config["calibration_protocol"]:
        raise SystemExit("Frozen evaluation/calibration protocols changed.")
    expected_indices = random.Random(20260920).sample(range(len(rows["train"])), 8192)
    if calibration["indices"] != expected_indices or protocol["conditions"] != robust.CONDITIONS:
        raise SystemExit("Unexpected calibration subset or corruption conditions.")
    if calibration["split"] != "train" or calibration["data_sha256"] != hashes or protocol["data_sha256"] != hashes:
        raise SystemExit("Calibration/data provenance mismatch.")
    with (robustness_run / "summary.csv").open(encoding="utf-8", newline="") as stream:
        reference = {row["condition"]: float(row["accuracy_pct"]) for row in csv.DictReader(stream)
                     if row["bn_variant"] == "calibrated_bn"}
    if set(reference) != {c["name"] for c in robust.CONDITIONS}:
        raise SystemExit("Incomplete original robustness results.")
    parent = ROOT / "outputs" / f"controls_{datetime.now():%Y%m%d_%H%M%S_%f}_seed{args.seed}"
    parent.mkdir(parents=True, exist_ok=False)
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.StreamHandler(), logging.FileHandler(ROOT / "logs" / f"{parent.name}.log", encoding="utf-8")])
    logging.info("Output: %s", parent)
    logging.info("GPU=%s CUDA_VISIBLE_DEVICES=%s", torch.cuda.get_device_name(0), os.getenv("CUDA_VISIBLE_DEVICES"))
    shutil.copyfile(Path(__file__), parent / "train_pilot_controls.py")
    shutil.copyfile(helper_path, parent / "original_train_pilot.py")
    shutil.copyfile(robust_path, parent / "eval_pilot_robustness.py")
    shutil.copyfile(robustness_run / "val_sample_index.csv", parent / "val_sample_index.csv")
    shared_config = {
        "args": vars(args), "data_config": data_config, "data_sha256": hashes,
        "reference_run": str(reference_run), "robustness_run": str(robustness_run),
        "reference_checkpoint_used_for_initialization": False,
        "source_sha256": helper.sha256(Path(__file__)), "helper_sha256": helper.sha256(helper_path),
        "robustness_source_sha256": helper.sha256(robust_path),
        "evaluation_protocol": protocol, "bn_calibration": calibration,
        "augmentation_seed": args.seed + 3000, "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__), "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "methods": list(METHODS), "training_corruptions": robust.CONDITIONS[1:],
        "interpretation": "known synthetic corruption distribution, writer-disjoint validation; one seed pilot",
    }
    write_json(parent / "config.json", shared_config)
    robust.check_transforms()
    results = {}
    for method in METHODS:
        results[method] = train_one(method, parent, args, rows, data_config, calibration,
                                    protocol, helper, robust, shared_config)
        gc.collect()
        torch.cuda.empty_cache()
    comparison = compare_results(reference, results["clean_pair"], results["aug_ce"], robust.CONDITIONS)
    write_csv(parent / "comparison.csv", comparison)
    write_json(parent / "result.json", {"completed": True, "test_evaluated": False,
               "comparison": comparison, "primary_contrast": "aug_ce minus clean_pair, calibrated BN, fixed epoch 10"})
    logging.info("DONE: %s", parent / "comparison.csv")
    print((parent / "comparison.csv").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
