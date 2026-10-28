"""Exploratory validation on eight predeclared, training-unseen corruptions.

Loads the existing calibrated epoch-10 controls. No training, recalibration,
checkpoint selection or use of the official test/confusion splits.
"""

import argparse
import html
import json
import logging
import os
import random
import shutil
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torchvision
import importlib.util
import sys

ROOT = Path("/workspace/shouxieti")
METHODS = ("clean_pair", "aug_ce")
CONDITIONS = [
    {"name": "clean", "category": "clean", "type": "identity"},
    {"name": "blur_s0.75", "category": "interpolation", "type": "gaussian", "sigma": 0.75, "kernel": 7},
    {"name": "blur_s1.25", "category": "interpolation", "type": "gaussian", "sigma": 1.25, "kernel": 9},
    {"name": "lowres_40", "category": "interpolation", "type": "lowres", "size": 40},
    {"name": "lowres_28", "category": "interpolation", "type": "lowres", "size": 28},
    {"name": "blur_s2.0", "category": "extrapolation", "type": "gaussian", "sigma": 2.0, "kernel": 13},
    {"name": "lowres_20", "category": "extrapolation", "type": "lowres", "size": 20},
    {"name": "lowres32_then_blur0.5", "category": "composition", "type": "compose", "operations": [
        {"type": "lowres", "size": 32}, {"type": "gaussian", "sigma": 0.5, "kernel": 5}]},
    {"name": "lowres32_then_blur1.0", "category": "composition", "type": "compose", "operations": [
        {"type": "lowres", "size": 32}, {"type": "gaussian", "sigma": 1.0, "kernel": 7}]},
]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def apply_condition(images, condition, base_corrupt):
    if condition["type"] == "compose":
        for operation in condition["operations"]:
            images = base_corrupt(images, operation)
        return images
    return base_corrupt(images, condition)


def summarize(predictions, targets):
    clean_ok = predictions[0] == targets
    clean_accuracy = float(clean_ok.mean())
    result = []
    for i, condition in enumerate(CONDITIONS):
        ok = predictions[i] == targets
        flips = int((clean_ok & ~ok).sum())
        result.append({"condition": condition["name"], "category": condition["category"],
                       "samples": len(targets), "accuracy_pct": 100 * float(ok.mean()),
                       "errors": int((~ok).sum()),
                       "drop_from_clean_pp": 100 * (clean_accuracy - float(ok.mean())),
                       "clean_correct_to_wrong": flips,
                       "clean_correct_to_wrong_pct": 100 * flips / max(1, int(clean_ok.sum())),
                       "clean_wrong_to_correct": int((~clean_ok & ok).sum())})
    return result


class ProtocolAdapter:
    CONDITIONS = CONDITIONS

    def __init__(self, base_corrupt):
        self.base_corrupt = base_corrupt

    def corrupt(self, images, condition):
        return apply_condition(images, condition, self.base_corrupt)

    @staticmethod
    def summarize(predictions, targets):
        return summarize(predictions, targets)


def make_comparison(predictions, targets):
    clean_teacher_ok = predictions["clean_pair"][0] == targets
    augmented_clean_ok = predictions["aug_ce"][0] == targets
    result = []
    for i, condition in enumerate(CONDITIONS):
        clean_ok = predictions["clean_pair"][i] == targets
        aug_ok = predictions["aug_ce"][i] == targets
        result.append({"condition": condition["name"], "category": condition["category"],
                       "clean_pair_pct": 100 * float(clean_ok.mean()),
                       "aug_ce_pct": 100 * float(aug_ok.mean()),
                       "aug_minus_clean_pair_pp": 100 * float(aug_ok.mean() - clean_ok.mean()),
                       "aug_errors": int((~aug_ok).sum()),
                       "aug_clean_correct_corrupt_wrong": int((augmented_clean_ok & ~aug_ok).sum()),
                       "clean_teacher_correct_aug_wrong": int((clean_teacher_ok & ~aug_ok).sum())})
    return result


def category_means(comparison):
    result = []
    for category in ("interpolation", "extrapolation", "composition", "all_8_unseen"):
        selected = [r for r in comparison if r["category"] != "clean" and
                    (category == "all_8_unseen" or r["category"] == category)]
        row = {"category": category, "conditions": len(selected)}
        for name in ("clean_pair_pct", "aug_ce_pct", "aug_minus_clean_pair_pp"):
            row[name] = sum(r[name] for r in selected) / len(selected)
        result.append(row)
    return result


@torch.no_grad()
def preview(out, dataset, labels, predictions, adapter, robust):
    selected = random.Random(20260921).sample(range(len(dataset)), 16)
    header = "".join(f'<th>{html.escape(c["name"])}</th>' for c in CONDITIONS)
    lines = []
    for index in selected:
        image, target = dataset[index]
        image = image.unsqueeze(0).cuda()
        cells = []
        for ci, condition in enumerate(CONDITIONS):
            guess = int(predictions[ci, index])
            color = "#187344" if guess == target else "#be2438"
            cells.append(f'<td><img src="{robust.png_uri(adapter.corrupt(image, condition))}"><br>'
                         f'<span style="color:{color}">{html.escape(labels[guess])}</span></td>')
        lines.append(f'<tr><th>{html.escape(labels[target])}<br><small>row {index}</small></th>'
                     + "".join(cells) + '</tr>')
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>未见退化预览</title>'
            '<style>body{font:16px system-ui;margin:24px;background:#f5f6f8}table{border-collapse:collapse;'
            'background:white}td,th{padding:7px;border:1px solid #ddd;text-align:center}img{width:80px;height:80px}'
            'small{font-size:11px;color:#666}</style><h1>固定随机抽样的 16 张验证图像</h1>'
            '<p>左列为真实标签，各图下方为增强模型预测。绿色正确，红色错误。'
            '组合退化依次执行缩小并恢复至 64×64、再模糊。请结合字形判断较强退化后笔画是否仍可辨认。</p>'
            '<table><tr><th>真实标签</th>' + header + '</tr>' + "".join(lines) + '</table></html>')
    (out / "transfer_preview.html").write_text(page, encoding="utf-8")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls-run", type=Path, required=True)
    options = parser.parse_args()
    run = options.controls_run.resolve()
    if not run.is_relative_to(ROOT.resolve()):
        raise SystemExit("Run directory must be inside /workspace/shouxieti.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("Expose exactly one allocated GPU with CUDA_VISIBLE_DEVICES.")
    shared = read_json(run / "config.json")
    if str(torch.__version__) != shared["torch"] or str(torchvision.__version__) != shared["torchvision"]:
        raise SystemExit("Use the original torch/torchvision versions.")
    helper_path = run / "original_train_pilot.py"
    helper = load_module("transfer_original_pilot", helper_path)
    controls_path, robust_path = run / "train_pilot_controls.py", run / "eval_pilot_robustness.py"
    for path, key in ((helper_path, "helper_sha256"), (controls_path, "source_sha256"),
                      (robust_path, "robustness_source_sha256")):
        if helper.sha256(path) != shared[key]:
            raise SystemExit(f"Source snapshot changed: {path}")
    controls = load_module("transfer_controls", controls_path)
    robust = load_module("transfer_robustness", robust_path)
    rows, data_config, hashes = helper.load_data(ROOT)
    if hashes != shared["data_sha256"]:
        raise SystemExit("Data/config fingerprints changed.")
    args = argparse.Namespace(**shared["args"])
    controls.seed_everything(args.seed, args.threads)
    out = ROOT / "outputs" / f"transfer_{datetime.now():%Y%m%d_%H%M%S_%f}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=False)
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.StreamHandler(), logging.FileHandler(ROOT / "logs" / f"{out.name}.log", encoding="utf-8")])
    logging.info("Output: %s", out)
    protocol = {"version": 1, "split": "val", "conditions": CONDITIONS,
                "status": "exploratory extension fixed before evaluating these conditions",
                "composition_order": "64->lowres->64 bilinear antialias, then Gaussian blur",
                "checkpoint": "existing calibrated epoch 10 for each method",
                "bn": "frozen saved training-only calibration; no adaptation",
                "precision": "FP32", "data_sha256": hashes,
                "scope": "training-unseen synthetic conditions on validation writers; not an official benchmark or test-set estimate"}
    robust.fixed_json(ROOT / "configs/pilot_transfer_v1.json", protocol)
    shutil.copyfile(Path(__file__), out / "eval_pilot_transfer.py")
    shutil.copyfile(run / "val_sample_index.csv", out / "val_sample_index.csv")
    targets = np.asarray([r[4] for r in rows["val"]], dtype=np.int64)
    models, original_clean_predictions, checkpoint_hashes = {}, {}, {}
    for method in METHODS:
        path = run / method / "epoch10_bn_calibrated.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        method_config = read_json(run / method / "config.json")
        if (checkpoint["epoch"] != 10 or checkpoint["config"] != method_config
                or method_config["method"] != method or checkpoint["bn_calibration"] != shared["bn_calibration"]
                or method_config["data_sha256"] != hashes or method_config["source_sha256"] != shared["source_sha256"]):
            raise SystemExit("Checkpoint/config/calibration mismatch.")
        model = helper.make_model()
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        models[method] = model
        checkpoint_hashes[method] = helper.sha256(path)
        del checkpoint
        with np.load(run / method / "predictions_calibrated_bn.npz", allow_pickle=False) as previous:
            if not np.array_equal(previous["targets"], targets) or str(previous["conditions"][0]) != "clean":
                raise SystemExit("Previous validation row order differs.")
            if not np.array_equal(previous["global_class_ids"], data_config["pilot_classes.json"]["global_class_ids"]):
                raise SystemExit("Previous class mapping differs.")
            original_clean_predictions[method] = previous["predictions"][0].copy()
    write_json = helper.write_json
    write_json(out / "config.json", {"protocol": protocol, "controls_run": str(run),
               "controls_config": shared, "checkpoint_sha256": checkpoint_hashes,
               "source_sha256": helper.sha256(Path(__file__)), "test_evaluated": False,
               "confusion_evaluated": False})
    adapter = ProtocolAdapter(robust.corrupt)
    white = torch.ones(2, 1, 64, 64, device="cuda")
    for condition in CONDITIONS:
        value = adapter.corrupt(white, condition)
        if value.shape != white.shape or not torch.allclose(value, white, atol=2e-6, rtol=0):
            raise RuntimeError("Transform check failed.")
    loader = controls.make_loader(helper, rows["val"], 128, args.workers, False, args.seed + 1)
    summary = controls.evaluate_conditions(models, loader, targets, adapter, out, data_config)
    summary = [{"method": r["bn_variant"], **{k: v for k, v in r.items() if k != "bn_variant"}} for r in summary]
    controls.write_csv(out / "summary.csv", summary)
    predictions = {}
    for method in METHODS:
        with np.load(out / f"predictions_{method}.npz", allow_pickle=False) as result:
            predictions[method] = result["predictions"].copy()
        if not np.array_equal(predictions[method][0], original_clean_predictions[method]):
            raise RuntimeError("Clean predictions failed to reproduce. Inspect before interpreting transfer results.")
    comparison = make_comparison(predictions, targets)
    means = category_means(comparison)
    controls.write_csv(out / "comparison.csv", comparison)
    controls.write_csv(out / "category_means.csv", means)
    dataset = helper.GNTDataset(rows["val"])
    try:
        preview(out, dataset, data_config["pilot_classes.json"]["classes"], predictions["aug_ce"], adapter, robust)
    finally:
        dataset.close()
    write_json(out / "result.json", {"completed": True, "clean_predictions_match_previous": True,
               "test_evaluated": False, "comparison": comparison, "category_means": means})
    logging.info("DONE: %s", out / "comparison.csv")
    print((out / "comparison.csv").read_text(encoding="utf-8"), flush=True)
    print((out / "category_means.csv").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
