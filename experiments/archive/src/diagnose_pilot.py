"""Replay the first six pilot epochs and isolate BatchNorm/last-batch effects.

The original run is read only. Only training images update BN statistics.
Diagnostic copies are discarded; the replay model keeps its original trajectory.
"""

import argparse
import copy
import csv
import importlib.util
import json
import logging
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.optim.swa_utils import update_bn
from torch.utils.data import DataLoader

ROOT = Path("/workspace/shouxieti")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_original(run_dir):
    source = run_dir / "train_pilot.py"
    spec = importlib.util.spec_from_file_location("original_pilot_snapshot", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def bn_buffers(model):
    suffixes = ("running_mean", "running_var", "num_batches_tracked")
    return {name: value.detach().clone() for name, value in model.named_buffers()
            if name.endswith(suffixes)}


@torch.no_grad()
def diagnose_model(model, before_bn, calibration_loader, val_loader, helper, amp, out, epoch):
    """All changes are made on a separate model copy, never the replay model."""
    criterion = nn.CrossEntropyLoss()
    clone = copy.deepcopy(model)
    post_fp32_loss, post_fp32_acc = helper.evaluate(clone, val_loader, criterion, False)
    # Keep post-tail weights, but substitute the pre-tail BN buffers.
    for name, value in clone.named_buffers():
        if name in before_bn:
            value.copy_(before_bn[name])
    restored_loss, restored_acc = helper.evaluate(clone, val_loader, criterion, amp)

    # ResNet18 has no dropout. update_bn resets BN buffers, then estimates them
    # on a fixed 8192-image subset of TRAIN, with no backward/optimizer step.
    update_bn(calibration_loader, clone, device=torch.device("cuda:0"))
    calibrated_loss, calibrated_acc = helper.evaluate(clone, val_loader, criterion, amp)
    calibrated_fp32_loss, calibrated_fp32_acc = helper.evaluate(clone, val_loader, criterion, False)
    if not all(torch.equal(a, b) for a, b in zip(model.parameters(), clone.parameters())):
        raise RuntimeError("Calibration unexpectedly changed a learned parameter.")
    helper.save_checkpoint(out / f"epoch_{epoch:02d}_bn_calibrated_diagnostic_only.pt", {
        "model": clone.state_dict(), "epoch": epoch, "diagnostic_only": True,
        "calibration_split": "train", "calibration_samples": len(calibration_loader.dataset),
        "val_accuracy": calibrated_acc,
    })
    del clone
    return {
        "post_tail_fp32_acc_pct": 100 * post_fp32_acc,
        "post_tail_fp32_loss": post_fp32_loss,
        "post_weights_pre_tail_bn_acc_pct": 100 * restored_acc,
        "post_weights_pre_tail_bn_loss": restored_loss,
        "calibrated_bn_acc_pct": 100 * calibrated_acc,
        "calibrated_bn_loss": calibrated_loss,
        "calibrated_bn_fp32_acc_pct": 100 * calibrated_fp32_acc,
        "calibrated_bn_fp32_loss": calibrated_fp32_loss,
        "learned_parameters_unchanged": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    opts = parser.parse_args()
    run_dir = opts.run_dir.resolve()
    if not run_dir.is_relative_to(ROOT.resolve()):
        raise SystemExit("The original run must be inside /workspace/shouxieti.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("Expose one allocated GPU using CUDA_VISIBLE_DEVICES.")

    original_config = read_json(run_dir / "config.json")
    args = argparse.Namespace(**original_config["args"])
    if args.mode != "pilot" or args.epochs < 6 or args.batch_size != 128:
        raise SystemExit("This diagnostic expects the original 128-batch, >=6-epoch pilot.")
    if str(torch.__version__) != original_config["torch"]:
        raise SystemExit("Use the same PyTorch version as the original run.")
    helper = load_original(run_dir)
    if helper.sha256(run_dir / "train_pilot.py") != original_config["source_sha256"]:
        raise SystemExit("The original source snapshot has changed.")
    rows, _, hashes = helper.load_data(ROOT)
    if hashes != original_config["data_sha256"]:
        raise SystemExit("Data/config fingerprints differ from the original run.")

    out = ROOT / "outputs" / f"diagnose_{datetime.now():%Y%m%d_%H%M%S_%f}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=False)
    log_path = ROOT / "logs" / f"{out.name}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")])
    logging.info("Output: %s", out)
    logging.info("GPU=%s CUDA_VISIBLE_DEVICES=%s", torch.cuda.get_device_name(0), os.getenv("CUDA_VISIBLE_DEVICES"))
    logging.info("Replay epochs 1..6; cosine schedule T_max=%d remains unchanged.", args.epochs)
    logging.info("Training samples=%d; batch_size=%d; final batch=%d.",
                 len(rows["train"]), args.batch_size, len(rows["train"]) % args.batch_size)
    shutil.copyfile(Path(__file__), out / "diagnose_pilot.py")
    shutil.copyfile(run_dir / "train_pilot.py", out / "original_train_pilot.py")
    with (run_dir / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        reference = {int(r["epoch"]): float(r["val_accuracy"]) for r in csv.DictReader(stream)}

    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = helper.make_model()
    loaders = {}
    for i, split in enumerate(("train", "val")):
        loaders[split] = DataLoader(
            helper.GNTDataset(rows[split]), batch_size=args.batch_size,
            shuffle=(split == "train"), num_workers=args.workers,
            pin_memory=True, persistent_workers=(args.workers > 0),
            worker_init_fn=helper.seed_worker,
            generator=torch.Generator().manual_seed(args.seed + i),
        )
    # Independent generators leave the original train/val sampler streams intact.
    diagnostic_val = DataLoader(
        helper.GNTDataset(rows["val"]), batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=(args.workers > 0),
        worker_init_fn=helper.seed_worker, generator=torch.Generator().manual_seed(args.seed + 1000),
    )
    calibration_indices = random.Random(20260920).sample(range(len(rows["train"])), 8192)
    calibration_loader = DataLoader(
        helper.GNTDataset([rows["train"][i] for i in calibration_indices]),
        batch_size=128, shuffle=False, num_workers=args.workers, pin_memory=True,
        persistent_workers=(args.workers > 0), worker_init_fn=helper.seed_worker,
        generator=torch.Generator().manual_seed(args.seed + 2000),
    )
    write_json(out / "calibration_train_indices.json", calibration_indices)
    write_json(out / "config.json", {
        "original_run": str(run_dir), "original_config": original_config,
        "replay_epochs": 6, "diagnostic_epochs": [2, 3, 6],
        "calibration_split": "train", "calibration_samples": 8192,
        "script_sha256": helper.sha256(Path(__file__)),
        "test_evaluated": False, "confusion_evaluated": False,
    })

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp = not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    diagnoses, replay = [], []
    for epoch in range(1, 7):
        model.train()
        loss_sum = correct = count = 0
        pre_tail_acc, before_bn = None, None
        lr = optimizer.param_groups[0]["lr"]
        for batch_index, (images, labels) in enumerate(loaders["train"], 1):
            if epoch in (2, 3, 6) and batch_index == len(loaders["train"]):
                _, pre_tail_acc = helper.evaluate(model, diagnostic_val, criterion, amp)
                before_bn = bn_buffers(model)
                helper.save_checkpoint(out / f"epoch_{epoch:02d}_before_tail.pt", {
                    "model": model.state_dict(), "epoch": epoch, "diagnostic_only": True,
                    "stage": "before_final_batch", "config": original_config,
                })
                logging.info("epoch=%d BEFORE last batch: val_acc=%.2f%%", epoch, 100 * pre_tail_acc)
                model.train()
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss.")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            n = labels.numel()
            loss_sum += loss.item() * n
            correct += (logits.argmax(1) == labels).sum().item()
            count += n
            if batch_index == 1 or batch_index % 200 == 0:
                logging.info("epoch=%d/6 batch=%d/%d loss=%.4f train_acc=%.2f%%",
                             epoch, batch_index, len(loaders["train"]), loss_sum / count, 100 * correct / count)
        val_loss, val_acc = helper.evaluate(model, loaders["val"], criterion, amp)
        scheduler.step()
        matches = abs(val_acc - reference[epoch]) < 1e-6
        record = {"epoch": epoch, "lr": lr, "train_loss": loss_sum / count,
                  "train_accuracy": correct / count, "val_loss": val_loss,
                  "val_accuracy": val_acc, "original_val_accuracy": reference[epoch],
                  "replay_matches_reference": matches}
        replay.append(record)
        write_json(out / "replay_metrics.json", replay)
        logging.info("REPLAY EPOCH %d: val_acc=%.2f%% original=%.2f%% matches=%s",
                     epoch, 100 * val_acc, 100 * reference[epoch], matches)
        if epoch in (2, 3, 6):
            helper.save_checkpoint(out / f"epoch_{epoch:02d}_after_tail.pt", {
                "model": model.state_dict(), "epoch": epoch, "val_accuracy": val_acc,
                "config": original_config, "diagnostic_only": True,
            })
            # Isolate diagnostic random-number use from subsequent replay epochs.
            rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
            try:
                diagnostic = diagnose_model(model, before_bn, calibration_loader, diagnostic_val,
                                            helper, amp, out, epoch)
            finally:
                random.setstate(rng[0])
                np.random.set_state(rng[1])
                torch.set_rng_state(rng[2])
                torch.cuda.set_rng_state_all(rng[3])
            row = {"epoch": epoch, "pre_tail_acc_pct": 100 * pre_tail_acc,
                   "post_tail_acc_pct": 100 * val_acc, **diagnostic,
                   "original_acc_pct": 100 * reference[epoch], "replay_matches_reference": matches}
            diagnoses.append(row)
            write_json(out / "diagnostics.json", diagnoses)
            logging.info("DIAG epoch=%d pre_tail=%.2f%% post_tail=%.2f%% post_fp32=%.2f%% old_bn=%.2f%% calibrated_bn=%.2f%%",
                         epoch, row["pre_tail_acc_pct"], row["post_tail_acc_pct"],
                         row["post_tail_fp32_acc_pct"], row["post_weights_pre_tail_bn_acc_pct"],
                         row["calibrated_bn_acc_pct"])

    columns = ["epoch", "pre_tail_acc_pct", "post_tail_acc_pct", "post_tail_fp32_acc_pct",
               "post_weights_pre_tail_bn_acc_pct", "calibrated_bn_acc_pct",
               "calibrated_bn_fp32_acc_pct", "original_acc_pct", "replay_matches_reference"]
    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(diagnoses)
    write_json(out / "result.json", {"completed": True,
               "all_replay_epochs_match": all(r["replay_matches_reference"] for r in replay),
               "summary": str(out / "summary.csv"), "test_evaluated": False})
    logging.info("DONE: %s", out / "summary.csv")
    print((out / "summary.csv").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
