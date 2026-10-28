"""HWDB1.1: 128-image fit check and fixed 500-class clean baseline.

Requires the outputs of prepare_data.py. Run inside /workspace/shouxieti.
Test/confusion images are never loaded by this script.
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import shutil
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models import resnet18


ROOT = Path("/workspace/shouxieti")
EXPECTED_PREPROCESS = {
    "canvas_size": 64,
    "content_max_side": 56,
    "resize": "bilinear",
    "background": 255,
    "polarity": "black_on_white",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data(root):
    names = ["classes.json", "pilot_classes.json", "writer_splits.json", "preprocess.json"]
    config = {name: read_json(root / "configs" / name) for name in names}
    if config["preprocess.json"] != EXPECTED_PREPROCESS:
        raise ValueError("Preprocess config differs from the agreed 64/56 pixel pipeline.")
    all_classes = config["classes.json"]["classes"]
    pilot = config["pilot_classes.json"]
    ids = pilot["global_class_ids"]
    if len(all_classes) != 3755 or len(ids) != 500 or len(set(ids)) != 500:
        raise ValueError("Expected 3755 global classes and 500 distinct pilot classes.")
    if pilot["classes"] != [all_classes[i] for i in ids]:
        raise ValueError("Pilot class mapping mismatch.")
    splits = config["writer_splits.json"]["splits"]
    expected_writers = {"train": 200, "confusion": 20, "val": 20, "test": 60}
    for split, count in expected_writers.items():
        if len(splits[split]) != count or len(set(splits[split])) != count:
            raise ValueError(f"Invalid writer list for {split}.")
    all_writers = [w for split in expected_writers for w in splits[split]]
    if len(set(all_writers)) != 300:
        raise ValueError("Writer overlap between splits.")

    mapping = {str(global_id): local_id for local_id, global_id in enumerate(ids)}
    rows = {"train": [], "val": []}
    writer_sets = {key: set(splits[key]) for key in rows}
    paths = {}
    index_path = root / "data/processed/hwdb1_1_index.csv"
    with index_path.open(encoding="utf-8", newline="") as stream:
        for record in csv.DictReader(stream):
            split = record["split"]
            if split not in rows or record["class_id"] not in mapping:
                continue
            if int(record["writer_id"]) not in writer_sets[split]:
                raise ValueError("Index writer does not match the saved split.")
            rel = record["gnt_path"]
            if rel not in paths:
                path = (root / rel).resolve()
                if not path.is_relative_to(root.resolve()) or not path.is_file():
                    raise ValueError(f"Invalid GNT path: {rel}")
                paths[rel] = str(path)
            offset, width, height = [int(record[k]) for k in ("pixel_offset", "width", "height")]
            if offset < 10 or min(width, height) < 1:
                raise ValueError("Invalid pixel offset or dimensions.")
            rows[split].append((paths[rel], offset, width, height, mapping[record["class_id"]]))
    for split, records in rows.items():
        if {record[4] for record in records} != set(range(500)):
            raise ValueError(f"{split} does not cover all 500 pilot classes.")
    audit = read_json(root / "outputs/data_audit.json")
    for split in rows:
        if len(rows[split]) != audit["pilot_samples_per_split"][split]:
            raise ValueError(f"{split} sample count differs from data_audit.json.")
    hashes = {name: sha256(root / "configs" / name) for name in names}
    hashes["hwdb1_1_index.csv"] = sha256(index_path)
    return rows, config, hashes


def center_image(image):
    width, height = image.size
    scale = 56 / max(width, height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    small = image.resize(size, Image.Resampling.BILINEAR)
    canvas = Image.new("L", (64, 64), 255)
    canvas.paste(small, ((64 - small.width) // 2, (64 - small.height) // 2))
    return canvas


class GNTDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows
        self.handles = OrderedDict()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, offset, width, height, label = self.rows[index]
        handle = self.handles.pop(path, None)
        if handle is None:
            handle = open(path, "rb")
        self.handles[path] = handle
        if len(self.handles) > 64:
            self.handles.popitem(last=False)[1].close()
        handle.seek(offset)
        pixels = handle.read(width * height)
        if len(pixels) != width * height:
            raise ValueError(f"Truncated pixels: {path}, offset={offset}")
        canvas = center_image(Image.frombytes("L", (width, height), pixels))
        tensor = torch.from_numpy(np.array(canvas, dtype=np.uint8, copy=True))
        tensor = tensor.unsqueeze(0).float().div_(127.5).sub_(1.0)
        return tensor, label

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()

    def __getstate__(self):
        return {"rows": self.rows, "handles": OrderedDict()}


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def make_model():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 500)
    return model.cuda()


@torch.no_grad()
def evaluate(model, loader, criterion, amp):
    model.eval()
    loss_sum = correct = count = 0
    for images, labels in loader:
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)
        n = labels.numel()
        loss_sum += loss.item() * n
        correct += (logits.argmax(1) == labels).sum().item()
        count += n
    return loss_sum / count, correct / count


def check_overfit(model, rows, args, out):
    chosen = random.Random(args.seed).sample(range(len(rows)), 128)
    data = GNTDataset([rows[i] for i in chosen])
    try:
        batch = [data[i] for i in range(len(data))]
    finally:
        data.close()
    images = torch.stack([item[0] for item in batch]).cuda()
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long, device="cuda")
    write_json(out / "sample_indices.json", chosen)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    accuracy = 0.0
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "train_mode_loss", "eval_loss", "eval_accuracy"])
        for step in range(1, args.steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            if step == 1 or step % 25 == 0 or step == args.steps:
                model.eval()
                with torch.no_grad():
                    logits = model(images)
                    eval_loss = criterion(logits, labels).item()
                    accuracy = (logits.argmax(1) == labels).float().mean().item()
                writer.writerow([step, loss.item(), eval_loss, accuracy])
                stream.flush()
                logging.info("step=%d/%d loss=%.4f eval_loss=%.4f fit_accuracy=%.2f%%",
                             step, args.steps, loss.item(), eval_loss, 100 * accuracy)
                if step >= 50 and accuracy >= 0.99:
                    break
    passed = accuracy >= 0.99
    write_json(out / "result.json", {"passed": passed, "samples": 128,
               "steps": step, "fit_accuracy": accuracy, "threshold": 0.99})
    logging.info("OVERFIT CHECK: %s. This is training-sample memorization, not validation accuracy.",
                 "PASS" if passed else "FAIL")
    if not passed:
        raise SystemExit("Fit check failed; inspect the log before running the pilot.")


def save_checkpoint(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def train_pilot(model, rows, args, out, run_config):
    loaders = {}
    for i, split in enumerate(("train", "val")):
        generator = torch.Generator().manual_seed(args.seed + i)
        loaders[split] = DataLoader(
            GNTDataset(rows[split]), batch_size=args.batch_size,
            shuffle=(split == "train"), num_workers=args.workers,
            pin_memory=True, persistent_workers=(args.workers > 0),
            worker_init_fn=seed_worker, generator=generator,
        )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)
    best_accuracy, best_epoch = -1.0, 0
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "lr", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "seconds"])
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            model.train()
            loss_sum = correct = count = 0
            lr = optimizer.param_groups[0]["lr"]
            for batch_index, (images, labels) in enumerate(loaders["train"], 1):
                images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=not args.no_amp):
                    logits = model(images)
                    loss = criterion(logits, labels)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite loss; training stopped.")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                n = labels.numel()
                loss_sum += loss.item() * n
                correct += (logits.argmax(1) == labels).sum().item()
                count += n
                if batch_index == 1 or batch_index % 100 == 0:
                    logging.info("epoch=%d/%d batch=%d/%d loss=%.4f train_acc=%.2f%%",
                                 epoch, args.epochs, batch_index, len(loaders["train"]),
                                 loss_sum / count, 100 * correct / count)
            val_loss, val_accuracy = evaluate(model, loaders["val"], criterion, not args.no_amp)
            seconds = time.perf_counter() - started
            writer.writerow([epoch, lr, loss_sum / count, correct / count, val_loss, val_accuracy, seconds])
            stream.flush()
            logging.info("EPOCH %d: train_loss=%.4f train_acc=%.2f%% val_loss=%.4f val_acc=%.2f%% seconds=%.1f",
                         epoch, loss_sum / count, 100 * correct / count, val_loss, 100 * val_accuracy, seconds)
            scheduler.step()
            checkpoint = {"model": model.state_dict(), "epoch": epoch,
                          "val_accuracy": val_accuracy, "config": run_config}
            save_checkpoint(out / "last.pt", checkpoint)
            if val_accuracy > best_accuracy:
                best_accuracy, best_epoch = val_accuracy, epoch
                save_checkpoint(out / "best.pt", checkpoint)
    write_json(out / "result.json", {"best_epoch": best_epoch,
               "best_val_accuracy": best_accuracy, "epochs": args.epochs,
               "test_evaluated": False})
    logging.info("DONE: best_epoch=%d best_val_accuracy=%.2f%% checkpoint=%s",
                 best_epoch, 100 * best_accuracy, out / "best.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["overfit", "pilot"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--no-amp", action="store_true", help="Use float32 for the pilot; overfit always uses float32.")
    args = parser.parse_args()
    if min(args.batch_size, args.epochs, args.steps, args.threads) < 1 or args.workers < 0:
        parser.error("Invalid counts.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("Expose exactly one allocated GPU using CUDA_VISIBLE_DEVICES before running.")
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

    run_name = f"{args.mode}_{datetime.now():%Y%m%d_%H%M%S_%f}_seed{args.seed}"
    out = ROOT / "outputs" / run_name
    out.mkdir(parents=True, exist_ok=False)
    (ROOT / "logs").mkdir(exist_ok=True)
    log_path = ROOT / "logs" / f"{run_name}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")])
    logging.info("Output: %s", out)
    logging.info("GPU: %s; CUDA_VISIBLE_DEVICES=%s", torch.cuda.get_device_name(0), os.getenv("CUDA_VISIBLE_DEVICES"))
    rows, data_config, hashes = load_data(ROOT)
    logging.info("Classes=500; train=%d; val=%d; no augmentation", len(rows["train"]), len(rows["val"]))
    source = Path(__file__).resolve()
    run_config = {
        "args": vars(args), "data_config": data_config, "data_sha256": hashes,
        "source_sha256": sha256(source), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "model": "ResNet18, random init, 1-channel 3x3 stride-1 stem, no maxpool, 500 outputs",
        "input_normalization": "pixel / 127.5 - 1.0; white=+1, black=-1",
        "selection": "validation accuracy; test and confusion images unused",
        "mixed_precision": args.mode == "pilot" and not args.no_amp,
        "train_samples": len(rows["train"]), "val_samples": len(rows["val"]),
    }
    write_json(out / "config.json", run_config)
    shutil.copyfile(source, out / "train_pilot.py")
    model = make_model()
    if args.mode == "overfit":
        check_overfit(model, rows["train"], args, out)
    else:
        train_pilot(model, rows, args, out, run_config)
    logging.info("Results: %s; log: %s", out / "result.json", log_path)


if __name__ == "__main__":
    main()
