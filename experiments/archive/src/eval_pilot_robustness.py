"""Evaluate the fixed epoch-10 pilot on clean and synthetic degraded validation images.

Report BOTH original and training-only recalibrated BN. No test images, no
corruption-specific BN adaptation, no optimizer updates, no checkpoint selection.
"""

import argparse
import base64
import copy
import csv
import html
import importlib.util
import io
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
from PIL import Image
import PIL
import torch
import torch.nn.functional as F
from torch.optim.swa_utils import update_bn
from torch.utils.data import DataLoader
import torchvision

ROOT = Path("/workspace/shouxieti")
CONDITIONS = [
    {"name": "clean", "type": "identity"},
    {"name": "blur_s0.5", "type": "gaussian", "sigma": 0.5, "kernel": 5},
    {"name": "blur_s1.0", "type": "gaussian", "sigma": 1.0, "kernel": 7},
    {"name": "blur_s1.5", "type": "gaussian", "sigma": 1.5, "kernel": 11},
    {"name": "lowres_48", "type": "lowres", "size": 48},
    {"name": "lowres_32", "type": "lowres", "size": 32},
    {"name": "lowres_24", "type": "lowres", "size": 24},
]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def fixed_json(path, value):
    if path.exists() and read_json(path) != value:
        raise RuntimeError(f"Existing protocol differs: {path}. Do not overwrite silently.")
    if not path.exists():
        write_json(path, value)


def load_original(run_dir):
    spec = importlib.util.spec_from_file_location("robustness_original_pilot", run_dir / "train_pilot.py")
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    return helper


def corrupt(images, condition):
    """N x 1 x 64 x 64 float tensors in [-1,1]; no quantization between steps."""
    if condition["type"] == "identity":
        return images
    if condition["type"] == "gaussian":
        radius = condition["kernel"] // 2
        axis = torch.arange(-radius, radius + 1, device=images.device, dtype=images.dtype)
        weights = torch.exp(-0.5 * (axis / condition["sigma"]).square())
        weights = weights / weights.sum()
        kernel = (weights[:, None] * weights[None, :]).reshape(1, 1, 2 * radius + 1, 2 * radius + 1)
        padded = F.pad(images, (radius, radius, radius, radius), mode="reflect")
        return F.conv2d(padded, kernel).clamp(-1, 1)
    if condition["type"] == "lowres":
        size = condition["size"]
        small = F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
        return F.interpolate(small, size=(64, 64), mode="bilinear", align_corners=False, antialias=True).clamp(-1, 1)
    raise ValueError(f"Unknown condition: {condition}")


@torch.no_grad()
def check_transforms():
    white = torch.ones(2, 1, 64, 64, device="cuda")
    pattern = white.clone()
    pattern[:, :, 24:40, 31:33] = -1
    for condition in CONDITIONS:
        result = corrupt(pattern, condition)
        if result.shape != pattern.shape or not torch.isfinite(result).all():
            raise RuntimeError(f"Invalid transform output: {condition['name']}")
        if result.min() < -1.00001 or result.max() > 1.00001:
            raise RuntimeError("Transform range failure.")
        if not torch.allclose(corrupt(white, condition), white, atol=2e-6, rtol=0):
            raise RuntimeError("Transform does not preserve a uniform white image.")
        if condition["type"] != "identity" and torch.equal(result, pattern):
            raise RuntimeError("Corruption unexpectedly equals the clean image.")
    logging.info("Transform shape/range/uniform-image checks: PASS")


def summarize(predictions, targets):
    clean_ok = predictions[0] == targets
    clean_acc = float(clean_ok.mean())
    results = []
    for i, condition in enumerate(CONDITIONS):
        ok = predictions[i] == targets
        flips = int((clean_ok & ~ok).sum())
        results.append({
            "condition": condition["name"], "samples": len(targets),
            "accuracy_pct": 100 * float(ok.mean()), "errors": int((~ok).sum()),
            "drop_from_clean_pp": 100 * (clean_acc - float(ok.mean())),
            "clean_correct_to_wrong": flips,
            "clean_correct_to_wrong_pct": 100 * flips / max(1, int(clean_ok.sum())),
            "clean_wrong_to_correct": int((~clean_ok & ok).sum()),
        })
    return results


def png_uri(tensor):
    pixels = tensor.detach().cpu().squeeze().add(1).mul(127.5).round().clamp(0, 255).numpy().astype(np.uint8)
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


@torch.no_grad()
def save_preview(out, dataset, classes, predictions):
    selected = random.Random(20260921).sample(range(len(dataset)), 16)
    header = "".join(f"<th>{html.escape(c['name'])}</th>" for c in CONDITIONS)
    lines = []
    for index in selected:
        tensor, label = dataset[index]
        images = tensor.unsqueeze(0).cuda()
        cells = []
        for j, condition in enumerate(CONDITIONS):
            shown = corrupt(images, condition)
            guess = int(predictions[j, index])
            color = "#187344" if guess == label else "#be2438"
            cells.append(f'<td><img src="{png_uri(shown)}"><br><span style="color:{color}">'
                         f'{html.escape(classes[guess])}</span></td>')
        lines.append(f'<tr><th>{html.escape(classes[label])}<br><small>row {index}</small></th>'
                     + "".join(cells) + '</tr>')
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>退化图像预览</title>'
            '<style>body{font:17px system-ui;margin:24px;background:#f5f6f8}table{border-collapse:collapse;'
            'background:white}td,th{padding:10px;border:1px solid #ddd;text-align:center}img{width:96px;height:96px}'
            'small{font-size:11px;color:#666}</style><h1>固定随机抽样的 16 张验证图像</h1>'
            '<p>左列为真实标签；各图下方为 BN 校准模型的预测。绿色表示正确，红色表示错误。'
            '模型输入为浮点图像；此处仅为显示而转换为 PNG。</p><table><tr><th>真实标签</th>'
            + header + '</tr>' + "".join(lines) + '</table></html>')
    (out / "corruption_preview.html").write_text(page, encoding="utf-8")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    opts = parser.parse_args()
    run_dir = opts.run_dir.resolve()
    if not run_dir.is_relative_to(ROOT.resolve()):
        raise SystemExit("The run directory must be inside /workspace/shouxieti.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("Expose exactly one allocated GPU with CUDA_VISIBLE_DEVICES.")
    original = read_json(run_dir / "config.json")
    args = argparse.Namespace(**original["args"])
    if args.mode != "pilot" or args.epochs != 10:
        raise SystemExit("This first robustness protocol expects the completed 10-epoch pilot.")
    if str(torch.__version__) != original["torch"] or str(torchvision.__version__) != original["torchvision"]:
        raise SystemExit("Use the same torch/torchvision versions as the original pilot.")
    helper = load_original(run_dir)
    if helper.sha256(run_dir / "train_pilot.py") != original["source_sha256"]:
        raise SystemExit("The original source snapshot has changed.")
    rows, data_config, hashes = helper.load_data(ROOT)
    if hashes != original["data_sha256"]:
        raise SystemExit("Data/config fingerprints differ from the original pilot.")

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

    out = ROOT / "outputs" / f"robustness_{datetime.now():%Y%m%d_%H%M%S_%f}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=False)
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.StreamHandler(), logging.FileHandler(ROOT / "logs" / f"{out.name}.log", encoding="utf-8")])
    logging.info("Output: %s", out)
    logging.info("GPU=%s CUDA_VISIBLE_DEVICES=%s", torch.cuda.get_device_name(0), os.getenv("CUDA_VISIBLE_DEVICES"))

    calibration_indices = random.Random(20260920).sample(range(len(rows["train"])), 8192)
    calibration_protocol = {"version": 1, "split": "train", "seed": 20260920,
                            "batch_size": 128, "indices": calibration_indices, "data_sha256": hashes}
    fixed_json(ROOT / "configs/bn_calibration_pilot_v1.json", calibration_protocol)
    protocol = {
        "version": 1, "split": "val", "conditions": CONDITIONS,
        "order": "GNT -> original aspect-preserving 64/56 canvas -> float normalization -> corruption",
        "gaussian": "normalized discrete separable kernel; radius=ceil(3*sigma); reflection padding",
        "lowres": "bilinear 64->size->64; align_corners=False; antialias=True for both steps",
        "quantization": "none after original uint8 canvas; float32 model inference",
        "bn": "raw and calibrated both reported; fixed 8192 clean training images, then frozen for all conditions",
        "checkpoint": "fixed epoch 10, last.pt; no additional checkpoint selection",
        "data_sha256": hashes,
    }
    fixed_json(ROOT / "configs/pilot_robustness_v1.json", protocol)
    shutil.copyfile(Path(__file__), out / "eval_pilot_robustness.py")
    shutil.copyfile(run_dir / "train_pilot.py", out / "original_train_pilot.py")
    write_json(out / "calibration_train_indices.json", calibration_indices)
    checkpoint_path = run_dir / "last.pt"
    # This is the user's own checkpoint, produced by the verified source snapshot.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["epoch"] != 10 or checkpoint["config"] != original:
        raise SystemExit("Checkpoint epoch/config does not match the requested original run.")
    raw_model = helper.make_model()
    raw_model.load_state_dict(checkpoint["model"], strict=True)
    raw_model.eval()
    calibrated_model = copy.deepcopy(raw_model)
    del checkpoint
    classes = data_config["pilot_classes.json"]["classes"]
    class_ids = data_config["pilot_classes.json"]["global_class_ids"]

    calibration_loader = DataLoader(
        helper.GNTDataset([rows["train"][i] for i in calibration_indices]),
        batch_size=128, shuffle=False, num_workers=args.workers, pin_memory=True,
        worker_init_fn=helper.seed_worker, generator=torch.Generator().manual_seed(args.seed + 2000))
    logging.info("Calibrating BN on 8192 fixed clean TRAIN images; learned parameters unchanged.")
    update_bn(calibration_loader, calibrated_model, device=torch.device("cuda:0"))
    calibrated_model.eval()
    if not all(torch.equal(a, b) for a, b in zip(raw_model.parameters(), calibrated_model.parameters())):
        raise RuntimeError("A learned parameter changed during BN calibration.")
    calibrated_config = {**original, "bn_calibration": calibration_protocol, "evaluation_protocol": protocol}
    helper.save_checkpoint(out / "epoch10_bn_calibrated.pt", {
        "model": calibrated_model.state_dict(), "epoch": 10, "config": calibrated_config,
        "source_checkpoint": str(checkpoint_path), "source_checkpoint_sha256": helper.sha256(checkpoint_path),
    })
    runtime = {"torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
               "numpy": np.__version__, "pillow": PIL.__version__, "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(0), "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES")}
    write_json(out / "config.json", {
        "protocol": protocol, "calibration_protocol": calibration_protocol,
        "original_run": str(run_dir), "original_config": original,
        "source_checkpoint_sha256": helper.sha256(checkpoint_path),
        "source_sha256": helper.sha256(Path(__file__)), "runtime": runtime,
        "test_evaluated": False, "confusion_evaluated": False,
    })
    check_transforms()

    dataset = helper.GNTDataset(rows["val"])
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=args.workers,
                        pin_memory=True, persistent_workers=(args.workers > 0),
                        worker_init_fn=helper.seed_worker,
                        generator=torch.Generator().manual_seed(args.seed + 1000))
    targets = np.asarray([r[4] for r in rows["val"]], dtype=np.int64)
    shape = (len(CONDITIONS), len(targets))
    models = {"raw_bn": raw_model, "calibrated_bn": calibrated_model}
    predictions = {name: np.empty(shape, dtype=np.int64) for name in models}
    confidences = {name: np.empty(shape, dtype=np.float32) for name in models}
    losses = {name: [] for name in models}
    initial_buffers = {name: {k: v.detach().clone() for k, v in model.named_buffers()} for name, model in models.items()}
    for ci, condition in enumerate(CONDITIONS):
        loss_sums = {name: 0.0 for name in models}
        offset = 0
        for images, labels in loader:
            images, labels_gpu = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            if not np.array_equal(labels.numpy(), targets[offset:offset + len(labels)]):
                raise RuntimeError("Validation row order changed.")
            transformed = corrupt(images, condition)
            for name, model in models.items():
                logits = model(transformed)
                if not torch.isfinite(logits).all():
                    raise RuntimeError(f"Non-finite logits: {name}, {condition['name']}")
                probability, predicted = logits.softmax(1).max(1)
                n = labels.numel()
                predictions[name][ci, offset:offset + n] = predicted.cpu().numpy()
                confidences[name][ci, offset:offset + n] = probability.cpu().numpy()
                loss_sums[name] += F.cross_entropy(logits, labels_gpu, reduction="sum").item()
            offset += labels.numel()
        if offset != len(targets):
            raise RuntimeError("Validation sample count mismatch.")
        for name in models:
            losses[name].append(loss_sums[name] / len(targets))
            accuracy = 100 * float((predictions[name][ci] == targets).mean())
            logging.info("EVAL %-13s %-12s accuracy=%.2f%% errors=%d/%d",
                         name, condition["name"], accuracy,
                         int((predictions[name][ci] != targets).sum()), len(targets))
    for name, model in models.items():
        if any(not torch.equal(v, initial_buffers[name][k]) for k, v in model.named_buffers()):
            raise RuntimeError("Model buffers changed during validation.")

    summary = []
    for name in models:
        values = summarize(predictions[name], targets)
        for i, row in enumerate(values):
            summary.append({"bn_variant": name, **row, "cross_entropy": losses[name][i]})
        np.savez_compressed(out / f"predictions_{name}.npz", predictions=predictions[name],
                            confidence=confidences[name], targets=targets,
                            conditions=np.asarray([c["name"] for c in CONDITIONS]),
                            global_class_ids=np.asarray(class_ids), classes=np.asarray(classes))
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (out / "val_sample_index.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row", "gnt_path", "pixel_offset", "width", "height", "writer_id", "local_class_id", "global_class_id", "character"])
        for i, (path, offset, width, height, label) in enumerate(rows["val"]):
            relative = Path(path).relative_to(ROOT).as_posix()
            writer.writerow([i, relative, offset, width, height, int(Path(path).stem.split('-')[0]), label, class_ids[label], classes[label]])
    try:
        save_preview(out, dataset, classes, predictions["calibrated_bn"])
    finally:
        dataset.close()
    write_json(out / "result.json", {"completed": True, "validation_samples": len(targets),
               "checkpoint_epoch": 10, "learned_parameters_unchanged": True,
               "bn_frozen_during_evaluation": True, "test_evaluated": False, "summary": summary})
    logging.info("DONE: %s", out / "summary.csv")
    print((out / "summary.csv").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
