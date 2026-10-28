"""Seed-42 500-class margin ablations and normalized cosine baselines. No test access."""
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


def fixed_margin_loss(scores, labels, selected=3):
    # Match the teacher-margin hard-negative selection exactly, including its pool.
    with torch.no_grad():
        ranking = scores.detach().clone()
        ranking.scatter_(1, labels[:, None], float("-inf"))
        pool = ranking.topk(min(10, scores.shape[1] - 1), dim=1).indices
        positions = scores.detach().gather(1, pool).topk(min(selected, pool.shape[1]), dim=1).indices
        rivals = pool.gather(1, positions)
    gap = scores.gather(1, labels[:, None]) - scores.gather(1, rivals)
    violation = F.relu(0.2 - gap)
    return violation.mean(), (violation.detach() > 0).any(1).float().mean()


METHODS = {
    "fixed_top1_degraded": {"head": "affine", "selected": 1, "views": "degraded", "margin": 0.2},
    "fixed_top3_both": {"head": "affine", "selected": 3, "views": "both", "margin": 0.2},
    "norm_softmax": {"head": "normalized", "margin": 0.0},
    "cosface_m010": {"head": "normalized", "margin": 0.1},
    "cosface_m020": {"head": "normalized", "margin": 0.2},
}
PROTOCOL = {
    "version": 1, "seed": 42, "classes": 500, "epochs": 10,
    "methods": METHODS, "normalized_scale": 30.0,
    "normalized_training": "L2 features and classifier weights, no bias, true-class additive cosine margin only in training CE; both views; from epoch 1",
    "normalized_inference": "30 * normalized features @ normalized classifier weights; NO true-label margin",
    "hinge": "mean relu(0.2 - (cos_y-cos_j)); pool 10; selected hardest 1 or 3; no teacher",
    "both_views_reduction": "mean across 2B views, NOT sum of two view means",
    "auxiliary_weight": "epoch 1: 0; epoch 2: 0.5; epochs 3-10: 1",
    "reuse": ["aug_ce seed42", "margin_fixed seed42 (top3 degraded only)"],
    "selection": "fixed epoch 10; report ALL methods; no automatic best-method selection",
    "interpretation": "exploratory validation only, single seed; corruption-unseen means absent in training, not an untouched evaluation set",
    "cosface_scope": "loss adaptation to HCCR, not reproduction of the face-recognition benchmark; scale and margins are bounded pilot choices, not claimed optimal",
    "cosface_source": "https://arxiv.org/abs/1801.09414",
    "test_evaluated": False, "confusion_evaluated": False,
}


def features_of(model, images):
    x = model.maxpool(model.relu(model.bn1(model.conv1(images))))
    x = model.layer4(model.layer3(model.layer2(model.layer1(x))))
    return torch.flatten(model.avgpool(x), 1)


class InferenceHead(torch.nn.Module):
    def __init__(self, backbone, normalized):
        super().__init__()
        self.backbone = backbone
        self.normalized = normalized

    def forward(self, images):
        features = features_of(self.backbone, images)
        if self.normalized:
            return 30.0 * (F.normalize(features.float(), dim=1) @
                           F.normalize(self.backbone.fc.weight.float(), dim=1).T)
        return self.backbone.fc(features)


def margin_logits(scores, labels, margin):
    return 30.0 * (scores - F.one_hot(labels, scores.shape[1]).to(scores.dtype) * margin)


def self_checks(helper, repeats):
    labels = torch.tensor([0], device="cuda")
    scores = torch.tensor([[0.1, 0.2, 0., -0.2]], device="cuda", requires_grad=True)
    loss, _ = fixed_margin_loss(scores, labels, 1)
    torch.testing.assert_close(loss, scores.new_tensor(0.3))
    loss.backward()
    torch.testing.assert_close(scores.grad, scores.new_tensor([[-1., 1., 0., 0.]]))
    scores.grad = None
    ours, _ = fixed_margin_loss(scores, labels, 3)
    old, _ = repeats.fixed_margin_loss(scores, labels)
    torch.testing.assert_close(ours, old, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(ours, scores, retain_graph=True)[0],
                               torch.autograd.grad(old, scores)[0], rtol=0, atol=0)
    doubled, _ = fixed_margin_loss(scores.repeat(2, 1), labels.repeat(2), 3)
    torch.testing.assert_close(doubled, ours.detach())
    torch.testing.assert_close(margin_logits(scores, labels, 0), scores * 30)
    expected = scores.detach().clone() * 30
    expected[0, 0] -= 6
    torch.testing.assert_close(margin_logits(scores, labels, .2), expected)
    model = helper.make_model().eval()
    x = torch.zeros(2, 1, 64, 64, device="cuda")
    with torch.no_grad():
        torch.testing.assert_close(InferenceHead(model, False)(x), model(x), rtol=0, atol=0)
        z = InferenceHead(model, True)(x)
        model.fc.bias.add_(100)
        torch.testing.assert_close(InferenceHead(model, True)(x), z, rtol=0, atol=0)
    f = torch.randn(2, 8, device="cuda", requires_grad=True)
    w = torch.randn(4, 8, device="cuda", requires_grad=True)
    ce = F.cross_entropy(margin_logits(F.normalize(f, dim=1) @ F.normalize(w, dim=1).T,
                                     torch.tensor([0, 1], device="cuda"), .2),
                         torch.tensor([0, 1], device="cuda"))
    ce.backward()
    require(bool(torch.isfinite(f.grad).all()) and bool(torch.isfinite(w.grad).all()), "Normalized gradient check failed")
    logging.info("SELF CHECKS PASS: top1/top3 losses/gradients; both-view mean; cosine margin; inference heads and bias exclusion")


def write_tables(parent, results, controls):
    detail, endpoints = [], []
    for method, suites in results.items():
        require(len(suites["seen"]) == 7 and len(suites["unseen"]) == 9, "Incomplete condition suite")
        require(abs(suites["seen"]["clean"] - suites["unseen"]["clean"]) < 1e-9,
                f"Clean predictions differ between suites: {method}")
        for suite, values in suites.items():
            for condition, value in values.items():
                detail.append({"method": method, "suite": suite, "condition": condition,
                    "accuracy_pct": value, "delta_vs_aug_ce_pp": value-results["aug_ce"][suite][condition],
                    "delta_vs_fixed_top3_degraded_pp": value-results["margin_fixed"][suite][condition]})
        endpoints.append({"method": method, "seed": 42, "clean_pct": suites["seen"]["clean"],
            "mean_of_6_seen_pct": float(np.mean([v for k,v in suites["seen"].items() if k != "clean"])),
            "mean_of_8_unseen_pct": float(np.mean([v for k,v in suites["unseen"].items() if k != "clean"])),
            "lowres_20_pct": suites["unseen"]["lowres_20"], "blur_s2.0_pct": suites["unseen"]["blur_s2.0"]})
    controls.write_csv(parent / "comparison.csv", detail)
    controls.write_csv(parent / "endpoints.csv", endpoints)


def train_one(method, parent, args, rows, data_config, shared, recipe, initial_hash, reference,
                helper, controls, robust, transfer, candidate, repeats):
    out = parent / method
    out.mkdir()
    controls.seed_everything(args.seed, args.threads)
    model = helper.make_model()
    require(controls.state_fingerprint(model) == initial_hash, "Fixed-margin initialization differs from baseline")
    spec = METHODS[method]
    normalized = spec["head"] == "normalized"
    if normalized:
        model.fc.bias.requires_grad_(False)
    prediction_model = InferenceHead(model, normalized)
    loader = controls.make_loader(helper, rows["train"], 128, args.workers, True, args.seed)
    val = controls.make_loader(helper, rows["val"], 128, args.workers, False, args.seed + 1)
    calibration = shared["bn_calibration"]
    cal_loader = controls.make_loader(helper, [rows["train"][i] for i in calibration["indices"]],
                                      128, args.workers, False, args.seed + 2000)
    generator = torch.Generator().manual_seed(args.seed + 3000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)
    config = {**recipe, "method": method, "inference_head": spec["head"], "inference_scale": 30 if normalized else None, "training_spec": spec, "initial_state_sha256": initial_hash,
              "teacher_checkpoint": None, "teacher_checkpoint_sha256": None}
    helper.write_json(out / "config.json", config)
    history = []
    counts_total = np.zeros(len(robust.CONDITIONS), dtype=np.int64)
    logging.info("START %s seed=%d; 10 epochs; spec=%s", method, args.seed, spec)
    for epoch in range(1, 11):
        started = time.perf_counter()
        model.train()
        n_total = skipped = clean_correct = corrupt_correct = 0
        ce_sum = aux_sum = active_sum = 0.0
        coefficient = 0.0 if normalized else candidate.auxiliary_weight(epoch)
        lr = optimizer.param_groups[0]["lr"]
        for step, (images, labels) in enumerate(loader, 1):
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            second, counts = controls.second_view(images, generator, robust)
            counts_total += counts
            n = len(labels)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=not args.no_amp):
                logits, features = candidate.forward_features(model, torch.cat((images, second)))
                if not normalized:
                    ce = F.cross_entropy(logits, torch.cat((labels, labels)))
            if normalized:
                scores = candidate.cosine_scores(model, features)
                logits = 30.0 * scores
                ce = F.cross_entropy(margin_logits(scores, torch.cat((labels, labels)), spec["margin"]),
                                     torch.cat((labels, labels)))
                auxiliary = active = ce.detach().new_zeros(())
            else:
                both = spec["views"] == "both"
                auxiliary, active = fixed_margin_loss(
                    candidate.cosine_scores(model, features if both else features[n:]),
                    torch.cat((labels, labels)) if both else labels, spec["selected"])
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
                logging.info("%s seed=%d epoch=%d/10 step=%d/%d CE=%.4f aux=%.4f weight=%.2f",
                             method, args.seed, epoch, step, len(loader), ce_sum / n_total, aux_sum / n_total, coefficient)
        train_seconds = time.perf_counter() - started
        raw_loss, raw_acc = helper.evaluate(prediction_model, val, torch.nn.CrossEntropyLoss(), False)
        calibrated = controls.calibrate_copy(prediction_model, cal_loader)
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
        logging.info("EPOCH %s seed=%d %d: raw=%.2f%% calibrated=%.2f%% seconds=%.1f skipped=%d",
                     method, args.seed, epoch, 100 * raw_acc, 100 * cal_acc, history[-1]["total_seconds"], skipped)
        if epoch == 1 and not normalized:
            matches = abs(raw_acc - reference["raw"]) < 1e-6 and abs(cal_acc - reference["calibrated"]) < 1e-6
            helper.write_json(out / "warmup_check.json", {"matches_aug_ce_epoch1": matches,
                              "raw_accuracy": raw_acc, "calibrated_accuracy": cal_acc, "reference": reference})
            require(matches, "CE-only epoch 1 did not reproduce the same-seed augmentation baseline")
        scheduler.step()
        helper.save_checkpoint(out / "last.pt", {"model": model.state_dict(), "epoch": epoch,
                               "config": config, "val_accuracy": raw_acc})
        if epoch != 10:
            del calibrated
    helper.save_checkpoint(out / "epoch10_bn_calibrated.pt", {"model": calibrated.backbone.state_dict(), "epoch": 10,
                           "config": config, "val_accuracy": cal_acc, "bn_calibration": calibration})
    metrics = repeats.evaluate_final(calibrated, out, rows, args, data_config, helper, controls, robust, transfer)
    helper.write_json(out / "result.json", {"completed": True, "method": method, "inference_head": spec["head"], "inference_scale": 30 if normalized else None, "training_spec": spec, "metrics": metrics,
                      "test_evaluated": False, "confusion_evaluated": False, "teacher_input_images": 0,
                      "student_input_images": sum(r["model_input_images"] for r in history),
                      "skipped_amp_steps": sum(r["skipped_amp_steps"] for r in history),
                      "second_view_counts": {c["name"]: int(counts_total[i]) for i, c in enumerate(robust.CONDITIONS)}})
    return metrics




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats-run", type=Path, required=True)
    opts = parser.parse_args()
    previous = opts.repeats_run.resolve()
    require(previous.is_relative_to(ROOT.resolve()), "Repeats run must be inside project")
    old = read_json(previous / "config.json")
    require(read_json(previous / "result.json")["completed"], "Repeats run incomplete")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0" and torch.cuda.is_available()
            and torch.cuda.device_count() == 1, "Set CUDA_VISIBLE_DEVICES=0")
    require(str(torch.__version__) == old["torch"] and str(torchvision.__version__) == old["torchvision"], "Environment changed")
    import hashlib
    sources = {name: (previous / Path(info["path"]).name, info["sha256"])
               for name, info in old["source_snapshots"].items()}
    sources["repeats"] = (previous / "train_pilot_margin_repeats.py", old["source_sha256"])
    for name, (path, digest) in sources.items():
        require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"Changed source: {name}")
    modules = {name: load_module(f"ablation_{name}", path) for name, (path, _) in sources.items()}
    helper, controls, robust, transfer, candidate, repeats = [modules[n] for n in
        ("helper", "controls", "robust", "transfer", "candidate", "repeats")]
    run, transfer_run = Path(old["controls_run"]).resolve(), Path(old["transfer_run"]).resolve()
    require(all(p.is_relative_to(ROOT.resolve()) for p in (run, transfer_run)), "Reference outside project")
    shared = read_json(run / "config.json")
    require(read_json(transfer_run / "config.json")["controls_config"] == shared, "Transfer reference changed")
    rows, data_config, hashes = helper.load_data(ROOT)
    require(hashes == old["data_sha256"] == shared["data_sha256"], "Data changed")
    require(len(data_config["pilot_classes.json"]["global_class_ids"]) == 500
            and len(rows["train"]) == 99594 and len(rows["val"]) == 9947, "Not original pilot")
    for file, expected in (("pilot_robustness_v1.json", old["seen_protocol"]),
                           ("pilot_transfer_v1.json", old["unseen_protocol"]),
                           ("bn_calibration_pilot_v1.json", old["bn_calibration"])):
        require(read_json(ROOT / "configs" / file) == expected, f"Protocol changed: {file}")
    require(old["bn_calibration"] == shared["bn_calibration"], "BN protocol changed")
    index_hash = helper.sha256(previous / "val_sample_index.csv")
    require(index_hash == helper.sha256(run / "val_sample_index.csv") == helper.sha256(transfer_run / "val_sample_index.csv"), "Sample order changed")
    args = argparse.Namespace(**shared["args"])
    require(args.seed == 42 and args.epochs == 10 and args.batch_size == 128, "Unexpected training recipe")
    parent = ROOT / "outputs" / f"margin_ablations_{datetime.now():%Y%m%d_%H%M%S_%f}_seed42"
    parent.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.StreamHandler(),
        logging.FileHandler(ROOT / "logs" / f"{parent.name}.log", encoding="utf-8")])
    logging.info("Output: %s; GPU=%s; 5 new training jobs", parent, torch.cuda.get_device_name(0))
    controls.seed_everything(42, args.threads)
    self_checks(helper, repeats)
    robust.fixed_json(ROOT / "configs/pilot_margin_ablations_v1.json", PROTOCOL)
    recipe = {"protocol": PROTOCOL, "args": vars(args), "data_config": data_config, "data_sha256": hashes,
              "repeats_run": str(previous), "controls_run": str(run), "transfer_run": str(transfer_run),
              "source_sha256": helper.sha256(Path(__file__)), "bn_calibration": old["bn_calibration"],
              "source_snapshots": {n: {"path": str(p), "sha256": h} for n,(p,h) in sources.items()},
              "seen_protocol": old["seen_protocol"], "unseen_protocol": old["unseen_protocol"],
              "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}
    helper.write_json(parent / "config.json", recipe)
    shutil.copyfile(Path(__file__), parent / Path(__file__).name)
    for path, _ in sources.values():
        shutil.copyfile(path, parent / path.name)
    shutil.copyfile(previous / "val_sample_index.csv", parent / "val_sample_index.csv")
    targets = np.asarray([r[4] for r in rows["val"]], dtype=np.int64)
    ids = data_config["pilot_classes.json"]["global_class_ids"]
    fixed = previous / "seed42/margin_fixed"
    require(read_json(fixed / "result.json")["completed"], "Fixed reference incomplete")
    require(read_json(fixed / "config.json")["data_sha256"] == hashes, "Fixed reference data changed")
    results = {"aug_ce": {}, "margin_fixed": {}}
    reused = []
    for suite, conditions in (("seen", robust.CONDITIONS), ("unseen", transfer.CONDITIONS)):
        paths = {"aug_ce": run / "aug_ce/predictions_calibrated_bn.npz" if suite == "seen" else transfer_run / "predictions_aug_ce.npz",
                 "margin_fixed": fixed / suite / "predictions_calibrated_bn.npz"}
        for method, path in paths.items():
            results[method][suite] = repeats.prediction_metrics(path, targets, [c["name"] for c in conditions], ids)
            reused.append({"method": method, "suite": suite, "path": str(path), "sha256": helper.sha256(path)})
    helper.write_json(parent / "reused_predictions.json", reused)
    write_tables(parent, results, controls)
    initial_hash = read_json(run / "initial_state.json")["sha256"]
    reference = repeats.warmup_reference(run / "aug_ce")
    for method in METHODS:
        results[method] = train_one(method, parent, args, rows, data_config, shared, recipe,
            initial_hash, reference, helper, controls, robust, transfer, candidate, repeats)
        write_tables(parent, results, controls)
        gc.collect()
        torch.cuda.empty_cache()
    helper.write_json(parent / "result.json", {"completed": True, "results": results,
        "training_seeds": [42], "new_training_jobs": 5, "test_evaluated": False, "confusion_evaluated": False})
    logging.info("DONE: %s", parent)
    print((parent / "endpoints.csv").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
