"""Pilot mechanism test: gated KD, hard-negative margin, degradation-delta margin.

This is an unvalidated online candidate, not an established novel method.
All students use the existing paired augmentation recipe and fixed epoch 10.
Only the 200-writer train split supplies optimization data; the teacher is frozen.
"""

import argparse
import csv
import gc
import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import importlib.util
import sys

ROOT = Path("/workspace/shouxieti")
METHODS = ("kd", "margin_hard", "margin_delta")
LOSS_PROTOCOL = {
    "version": 1,
    "status": "exploratory online mechanism test; no claim of confirmed novelty or effectiveness",
    "teacher": "frozen clean_pair calibrated epoch 10; clean training input only; FP32",
    "student": "identical random initialization and paired CE recipe as aug_ce",
    "kd": {"temperature": 2.0, "direction": "KL(teacher clean || student corrupted)",
           "gate": "teacher clean prediction equals training label", "scale": "T squared"},
    "margin": {"scores": "cosine between pooled feature and classifier weight; classifier bias excluded",
               "pool": 10, "selected": 3, "target_cap": 0.2,
               "teacher_target": "clip(teacher cosine y minus cosine competitor, 0, cap)",
               "gate": "teacher prediction correct AND teacher target strictly positive",
               "hard": "top 3 corrupted cosine competitors among the top-10 incorrect classes",
               "delta": "top 3 positive increases in competitor-relative-to-true score within the same pool",
               "delta_definition": "(corrupt[j]-corrupt[y]) - (clean[j]-clean[y]); detached selection",
               "reduction": "mean over eligible pairs per sample, then mean over ALL source samples"},
    "auxiliary_weight": {"epoch_1": 0.0, "epoch_2": 0.5, "epochs_3_to_10": 1.0},
    "selection": "fixed epoch 10; no selection by validation maxima",
    "primary_pilot_endpoint": "mean accuracy over the frozen eight unseen conditions",
    "secondary_endpoints": "clean and mean of the original six known corruptions; all individual conditions reported",
    "data": "same train split as aug_ce; confusion and test splits unused",
    "compute": "same student source batches and paired views; extra clean teacher inference is logged, not claimed free",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def forward_features(model, x):
    """The same ResNet18 forward path, also exposing its pooled features."""
    x = model.relu(model.bn1(model.conv1(x)))
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    features = torch.flatten(model.avgpool(x), 1)
    return model.fc(features), features


def cosine_scores(model, features):
    # Called outside autocast so the auxiliary geometry is evaluated in FP32.
    return F.normalize(features.float(), dim=1) @ F.normalize(model.fc.weight.float(), dim=1).t()


def kd_loss(student_logits, teacher_logits, labels, temperature=2.0):
    teacher_logits = teacher_logits.detach().float()
    gate = teacher_logits.argmax(1).eq(labels)
    target = F.softmax(teacher_logits / temperature, dim=1)
    values = F.kl_div(F.log_softmax(student_logits.float() / temperature, dim=1),
                      target, reduction="none").sum(1) * temperature ** 2
    return (values * gate.float()).mean(), gate.float().mean()


def margin_loss(clean_scores, corrupt_scores, teacher_scores, teacher_correct, labels,
                use_delta, pool_size=10, selected=3, cap=0.2):
    pool_size = min(pool_size, corrupt_scores.shape[1] - 1)
    selected = min(selected, pool_size)
    with torch.no_grad():
        clean = clean_scores.detach()
        degraded = corrupt_scores.detach()
        pool_rank = degraded.clone()
        pool_rank.scatter_(1, labels[:, None], float("-inf"))
        pool = pool_rank.topk(pool_size, dim=1).indices
        relative_clean = clean.gather(1, pool) - clean.gather(1, labels[:, None])
        relative_corrupt = degraded.gather(1, pool) - degraded.gather(1, labels[:, None])
        delta = relative_corrupt - relative_clean
        ranking = delta if use_delta else degraded.gather(1, pool)
        positions = ranking.topk(selected, dim=1).indices
        rivals = pool.gather(1, positions)
        teacher = teacher_scores.detach()
        target = (teacher.gather(1, labels[:, None]) - teacher.gather(1, rivals)).clamp(0, cap)
        eligible = teacher_correct[:, None] & (target > 0)
        if use_delta:
            eligible = eligible & (delta.gather(1, positions) > 0)
    actual = corrupt_scores.gather(1, labels[:, None]) - corrupt_scores.gather(1, rivals)
    violations = F.relu(target - actual)
    per_sample = (violations * eligible.float()).sum(1) / eligible.sum(1).clamp_min(1)
    active = ((violations.detach() > 0) & eligible).any(1).float().mean()
    return per_sample.mean(), active


def auxiliary_weight(epoch):
    return min(1.0, max(0.0, (epoch - 1) / 2.0))


def self_checks(teacher):
    """Analytical loss/gradient cases run before any long training job."""
    labels = torch.tensor([0], device="cuda")
    correct = torch.tensor([True], device="cuda")
    clean = torch.tensor([[0.9, 0.8, 0.0, -0.2]], device="cuda")
    teacher_cos = torch.tensor([[0.8, 0.3, 0.1, 0.0]], device="cuda")
    degraded = torch.tensor([[0.1, 0.2, 0.05, -0.2]], device="cuda", requires_grad=True)
    delta, _ = margin_loss(clean, degraded, teacher_cos, correct, labels, True,
                           pool_size=3, selected=1, cap=0.2)
    hard, _ = margin_loss(clean, degraded, teacher_cos, correct, labels, False,
                          pool_size=3, selected=1, cap=0.2)
    torch.testing.assert_close(delta, torch.tensor(0.15, device="cuda"), atol=1e-6, rtol=0)
    torch.testing.assert_close(hard, torch.tensor(0.30, device="cuda"), atol=1e-6, rtol=0)
    delta.backward()
    torch.testing.assert_close(degraded.grad, torch.tensor([[-1.0, 0.0, 1.0, 0.0]], device="cuda"), atol=1e-6, rtol=0)
    gated, _ = margin_loss(clean, degraded.detach(), teacher_cos, ~correct, labels, True)
    unchanged, _ = margin_loss(clean, clean, teacher_cos, correct, labels, True)
    if gated.item() != 0 or unchanged.item() != 0:
        raise RuntimeError("Teacher gate or zero-delta test failed.")
    logits = torch.tensor([[2.0, -0.5, 0.3, -1.0]], device="cuda")
    identity_kd, _ = kd_loss(logits, logits, labels)
    if abs(identity_kd.item()) > 1e-5:
        raise RuntimeError("Identical-distribution KD should be zero.")
    with torch.no_grad():
        probe = torch.linspace(-1, 1, 2 * 64 * 64, device="cuda").reshape(2, 1, 64, 64)
        direct = teacher(probe)
        exposed, _ = forward_features(teacher, probe)
        torch.testing.assert_close(exposed, direct, atol=0, rtol=0)
    logging.info("SELF CHECKS PASS: margin values, gradient direction, teacher gate, zero-delta case, KD identity, ResNet forward equivalence.")


def train_one(method, parent, args, rows, data_config, teacher, helper, controls, robust, transfer,
              shared, recipe, original_initial_hash, warmup_reference):
    out = parent / method
    out.mkdir(exist_ok=False)
    controls.seed_everything(args.seed, args.threads)
    model = helper.make_model()
    initial_hash = controls.state_fingerprint(model)
    if initial_hash != original_initial_hash:
        raise RuntimeError("Student initialization differs from the aug_ce control.")
    train_loader = controls.make_loader(helper, rows["train"], 128, args.workers, True, args.seed)
    val_loader = controls.make_loader(helper, rows["val"], 128, args.workers, False, args.seed + 1)
    calibration = shared["bn_calibration"]
    calibration_loader = controls.make_loader(helper, [rows["train"][i] for i in calibration["indices"]],
                                              128, args.workers, False, args.seed + 2000)
    aug_generator = torch.Generator().manual_seed(args.seed + 3000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)
    config = {**recipe, "method": method, "initial_state_sha256": initial_hash}
    helper.write_json(out / "config.json", config)
    teacher_initial_hash = controls.state_fingerprint(teacher)
    history, final_calibrated = [], None
    condition_counts = np.zeros(len(robust.CONDITIONS), dtype=np.int64)
    logging.info("START %s: 10 epochs; paired source batch 128; frozen clean teacher.", method)
    for epoch in range(1, 11):
        started = time.perf_counter()
        model.train()
        coefficient = auxiliary_weight(epoch)
        sums = {k: 0.0 for k in ("ce", "aux", "loss", "active", "teacher_correct")}
        count = clean_correct = second_correct = skipped = 0
        lr = optimizer.param_groups[0]["lr"]
        for step, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            second, counts = controls.second_view(images, aug_generator, robust)
            condition_counts += counts
            n = labels.numel()
            with torch.no_grad():
                teacher_logits, teacher_features = forward_features(teacher, images)
                teacher_correct = teacher_logits.argmax(1).eq(labels)
                teacher_cos = cosine_scores(teacher, teacher_features) if method != "kd" else None
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=not args.no_amp):
                logits, features = forward_features(model, torch.cat((images, second), dim=0))
                ce = F.cross_entropy(logits, torch.cat((labels, labels), dim=0))
            if method == "kd":
                auxiliary, active = kd_loss(logits[n:], teacher_logits, labels)
            else:
                cosine = cosine_scores(model, features)
                auxiliary, active = margin_loss(cosine[:n], cosine[n:], teacher_cos, teacher_correct,
                                                labels, use_delta=(method == "margin_delta"))
            # Epoch 1 is a strict CE-only implementation check; do not backprop a zero-weight aux graph.
            loss = ce if coefficient == 0 else ce + coefficient * auxiliary
            if not torch.isfinite(loss) or not torch.isfinite(auxiliary):
                raise RuntimeError("Non-finite loss.")
            previous_scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            skipped += int(scaler.get_scale() < previous_scale)
            count += n
            sums["ce"] += ce.item() * n
            sums["aux"] += auxiliary.item() * n
            sums["loss"] += loss.item() * n
            sums["active"] += active.item() * n
            sums["teacher_correct"] += teacher_correct.float().mean().item() * n
            clean_correct += (logits[:n].argmax(1) == labels).sum().item()
            second_correct += (logits[n:].argmax(1) == labels).sum().item()
            if step == 1 or step % 200 == 0:
                logging.info("%s epoch=%d/10 step=%d/%d CE=%.4f aux=%.4f weight=%.2f clean_acc=%.2f%% corrupt_acc=%.2f%%",
                             method, epoch, step, len(train_loader), sums["ce"] / count, sums["aux"] / count,
                             coefficient, 100 * clean_correct / count, 100 * second_correct / count)
        train_seconds = time.perf_counter() - started
        raw_loss, raw_acc = helper.evaluate(model, val_loader, torch.nn.CrossEntropyLoss(), False)
        calibrated = controls.calibrate_copy(model, calibration_loader)
        calibrated_loss, calibrated_acc = helper.evaluate(calibrated, val_loader, torch.nn.CrossEntropyLoss(), False)
        row = {"epoch": epoch, "lr": lr, "auxiliary_weight": coefficient,
               **{f"train_{key}": value / count for key, value in sums.items()},
               "train_clean_accuracy": clean_correct / count, "train_corrupt_accuracy": second_correct / count,
               "val_raw_loss": raw_loss, "val_raw_accuracy": raw_acc,
               "val_calibrated_loss": calibrated_loss, "val_calibrated_accuracy": calibrated_acc,
               "train_source_images": count, "model_input_images": 2 * count, "teacher_input_images": count,
               "attempted_steps": len(train_loader), "skipped_amp_steps": skipped,
               "train_seconds": train_seconds, "total_seconds": time.perf_counter() - started}
        history.append(row)
        controls.write_csv(out / "metrics.csv", history)
        logging.info("EPOCH %s %d: calibrated_val=%.2f%% raw_val=%.2f%% aux=%.5f active=%.2f%% seconds=%.1f skipped=%d",
                     method, epoch, 100 * calibrated_acc, 100 * raw_acc, sums["aux"] / count,
                     100 * sums["active"] / count, row["total_seconds"], skipped)
        if epoch == 1:
            matches = (abs(raw_acc - warmup_reference["raw"]) < 1e-6
                       and abs(calibrated_acc - warmup_reference["calibrated"]) < 1e-6)
            helper.write_json(out / "warmup_check.json", {"matches_aug_ce_epoch1": matches,
                              "raw_accuracy": raw_acc, "calibrated_accuracy": calibrated_acc,
                              "reference": warmup_reference})
            if not matches:
                raise RuntimeError("CE-only epoch 1 did not reproduce aug_ce. Inspect before continuing.")
        scheduler.step()
        helper.save_checkpoint(out / "last.pt", {"model": model.state_dict(), "epoch": epoch,
                               "config": config, "val_accuracy": raw_acc})
        if epoch == 10:
            final_calibrated = calibrated
            helper.save_checkpoint(out / "epoch10_bn_calibrated.pt", {
                "model": calibrated.state_dict(), "epoch": epoch, "config": config,
                "val_accuracy": calibrated_acc, "bn_calibration": calibration})
        else:
            del calibrated
    if controls.state_fingerprint(teacher) != teacher_initial_hash:
        raise RuntimeError("Frozen teacher parameters or buffers changed.")
    targets = np.asarray([r[4] for r in rows["val"]], dtype=np.int64)
    metrics = {}
    for suite, adapter in (("seen", robust), ("unseen", transfer.ProtocolAdapter(robust.corrupt))):
        eval_dir = out / suite
        eval_dir.mkdir()
        values = controls.evaluate_conditions({"raw_bn": model, "calibrated_bn": final_calibrated},
                                              val_loader, targets, adapter, eval_dir, data_config)
        metrics[suite] = {r["condition"]: r["accuracy_pct"] for r in values if r["bn_variant"] == "calibrated_bn"}
    helper.write_json(out / "result.json", {"completed": True, "method": method,
                      "teacher_unchanged": True, "test_evaluated": False, "confusion_evaluated": False,
                      "attempted_steps": sum(r["attempted_steps"] for r in history),
                      "skipped_amp_steps": sum(r["skipped_amp_steps"] for r in history),
                      "student_input_images": 2 * sum(r["train_source_images"] for r in history),
                      "teacher_input_images": sum(r["teacher_input_images"] for r in history),
                      "second_view_counts": {c["name"]: int(condition_counts[i]) for i, c in enumerate(robust.CONDITIONS)},
                      "metrics": metrics})
    return metrics


def comparison_rows(reference, results, conditions, suite):
    table = []
    for condition in conditions:
        name = condition["name"]
        row = {"condition": name, "aug_ce_pct": reference[name]}
        for method, scores in results.items():
            value = scores[suite][name]
            row[f"{method}_pct"] = value
            row[f"{method}_minus_aug_ce_pp"] = value - reference[name]
        table.append(row)
    names = [c["name"] for c in conditions[1:]]
    row = {"condition": f"mean_of_{len(names)}_{suite}", "aug_ce_pct": sum(reference[n] for n in names) / len(names)}
    for method, scores in results.items():
        value = sum(scores[suite][n] for n in names) / len(names)
        row[f"{method}_pct"] = value
        row[f"{method}_minus_aug_ce_pp"] = value - row["aug_ce_pct"]
    return table + [row]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls-run", type=Path, required=True)
    parser.add_argument("--transfer-run", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    opts = parser.parse_args()
    if len(set(opts.methods)) != len(opts.methods):
        parser.error("Duplicate methods.")
    run, transfer_run = opts.controls_run.resolve(), opts.transfer_run.resolve()
    if any(not p.is_relative_to(ROOT.resolve()) for p in (run, transfer_run)):
        raise SystemExit("Run directories must be inside /workspace/shouxieti.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("Expose exactly one allocated GPU.")
    shared = read_json(run / "config.json")
    transfer_config = read_json(transfer_run / "config.json")
    if transfer_config["controls_config"] != shared or not read_json(transfer_run / "result.json")["clean_predictions_match_previous"]:
        raise SystemExit("Transfer results do not match the controls.")
    args = argparse.Namespace(**shared["args"])
    if args.epochs != 10 or args.batch_size != 128 or args.mode != "pilot":
        raise SystemExit("Expected the original 10-epoch batch-128 pilot recipe.")
    if str(torch.__version__) != shared["torch"] or str(torchvision.__version__) != shared["torchvision"]:
        raise SystemExit("Use the same torch/torchvision versions.")
    helper_path = run / "original_train_pilot.py"
    helper = load_module("candidate_original_pilot", helper_path)
    controls_path, robust_path = run / "train_pilot_controls.py", run / "eval_pilot_robustness.py"
    transfer_path = transfer_run / "eval_pilot_transfer.py"
    for path, expected in ((helper_path, shared["helper_sha256"]), (controls_path, shared["source_sha256"]),
                           (robust_path, shared["robustness_source_sha256"]), (transfer_path, transfer_config["source_sha256"])):
        if helper.sha256(path) != expected:
            raise SystemExit(f"Source snapshot changed: {path}")
    controls = load_module("candidate_controls", controls_path)
    robust = load_module("candidate_robustness", robust_path)
    transfer = load_module("candidate_transfer", transfer_path)
    rows, data_config, hashes = helper.load_data(ROOT)
    if hashes != shared["data_sha256"]:
        raise SystemExit("Data/config fingerprints changed.")
    if (read_json(ROOT / "configs/pilot_robustness_v1.json") != shared["evaluation_protocol"]
            or read_json(ROOT / "configs/pilot_transfer_v1.json") != transfer_config["protocol"]
            or read_json(ROOT / "configs/bn_calibration_pilot_v1.json") != shared["bn_calibration"]):
        raise SystemExit("Evaluation or calibration protocols changed.")
    controls.seed_everything(args.seed, args.threads)
    parent = ROOT / "outputs" / f"candidates_{datetime.now():%Y%m%d_%H%M%S_%f}_seed{args.seed}"
    parent.mkdir(parents=True, exist_ok=False)
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.StreamHandler(),
        logging.FileHandler(ROOT / "logs" / f"{parent.name}.log", encoding="utf-8")])
    logging.info("Output: %s", parent)
    logging.info("GPU=%s CUDA_VISIBLE_DEVICES=%s", torch.cuda.get_device_name(0), os.getenv("CUDA_VISIBLE_DEVICES"))
    robust.fixed_json(ROOT / "configs/pilot_candidate_losses_v1.json", LOSS_PROTOCOL)
    teacher_path = run / "clean_pair/epoch10_bn_calibrated.pt"
    checkpoint = torch.load(teacher_path, map_location="cpu", weights_only=False)
    if (checkpoint["epoch"] != 10 or checkpoint["config"] != read_json(run / "clean_pair/config.json")
            or checkpoint["bn_calibration"] != shared["bn_calibration"]):
        raise SystemExit("Teacher checkpoint mismatch.")
    teacher = helper.make_model()
    teacher.load_state_dict(checkpoint["model"], strict=True)
    teacher.requires_grad_(False)
    teacher.eval()
    del checkpoint
    self_checks(teacher)
    with (run / "aug_ce/metrics.csv").open(encoding="utf-8", newline="") as stream:
        first = next(csv.DictReader(stream))
    warmup_reference = {"raw": float(first["val_raw_accuracy"]), "calibrated": float(first["val_calibrated_accuracy"])}
    with (run / "aug_ce/summary.csv").open(encoding="utf-8", newline="") as stream:
        reference_seen = {r["condition"]: float(r["accuracy_pct"]) for r in csv.DictReader(stream) if r["bn_variant"] == "calibrated_bn"}
    with (transfer_run / "comparison.csv").open(encoding="utf-8", newline="") as stream:
        reference_unseen = {r["condition"]: float(r["aug_ce_pct"]) for r in csv.DictReader(stream)}
    recipe = {"args": vars(args), "data_config": data_config, "data_sha256": hashes,
              "controls_run": str(run), "transfer_run": str(transfer_run),
              "teacher_checkpoint": str(teacher_path), "teacher_checkpoint_sha256": helper.sha256(teacher_path),
              "loss_protocol": LOSS_PROTOCOL, "bn_calibration": shared["bn_calibration"],
              "seen_protocol": shared["evaluation_protocol"], "unseen_protocol": transfer_config["protocol"],
              "source_sha256": helper.sha256(Path(__file__)), "methods": opts.methods,
              "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}
    helper.write_json(parent / "config.json", recipe)
    for source in (Path(__file__), helper_path, controls_path, robust_path, transfer_path):
        shutil.copyfile(source, parent / source.name)
    shutil.copyfile(run / "val_sample_index.csv", parent / "val_sample_index.csv")
    initial_hash = read_json(run / "initial_state.json")["sha256"]
    results = {}
    for method in opts.methods:
        results[method] = train_one(method, parent, args, rows, data_config, teacher, helper, controls,
                                    robust, transfer, shared, recipe, initial_hash, warmup_reference)
        for suite, reference, conditions in (("seen", reference_seen, robust.CONDITIONS),
                                               ("unseen", reference_unseen, transfer.CONDITIONS)):
            table = comparison_rows(reference, results, conditions, suite)
            controls.write_csv(parent / f"comparison_{suite}.csv", table)
        gc.collect()
        torch.cuda.empty_cache()
    helper.write_json(parent / "result.json", {"completed": True, "methods": opts.methods,
                      "test_evaluated": False, "confusion_evaluated": False, "results": results})
    logging.info("DONE: %s", parent)
    print((parent / "comparison_seen.csv").read_text(encoding="utf-8"), flush=True)
    print((parent / "comparison_unseen.csv").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
