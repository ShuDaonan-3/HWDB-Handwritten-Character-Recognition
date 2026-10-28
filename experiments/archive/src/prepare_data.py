import base64
import csv
import html
import io
import json
import random
import struct
from collections import Counter
from pathlib import Path
from PIL import Image

ROOT = Path("/workspace/shouxieti")
SEED = 42


def records(stream, total, name):
    while stream.tell() < total:
        start = stream.tell()
        header = stream.read(10)
        if len(header) != 10:
            raise ValueError(f"{name}: incomplete header at {start}")

        size, tag, width, height = struct.unpack("<I2sHH", header)
        if width == 0 or height == 0 or size != 10 + width * height:
            raise ValueError(f"{name}: invalid record at {start}")
        if start + size > total:
            raise ValueError(f"{name}: truncated image at {start}")

        yield tag, width, height, start + 10
        stream.seek(start + size)


def fixed_json(name, value):
    path = ROOT / "configs" / name
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != value:
            raise RuntimeError(f"Existing config differs: {path}")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def png_url(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(
        buf.getvalue()
    ).decode("ascii")


def make_preview(samples):
    cards = []
    for relpath, offset, width, height, cid, writer, label in samples:
        with (ROOT / relpath).open("rb") as f:
            f.seek(offset)
            pixels = f.read(width * height)
        if len(pixels) != width * height:
            raise ValueError(f"Could not read preview: {relpath}:{offset}")

        image = Image.frombytes("L", (width, height), pixels)
        scale = 56 / max(width, height)
        small = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.BILINEAR,
        )
        padded = Image.new("L", (64, 64), 255)
        padded.paste(
            small,
            ((64 - small.width) // 2, (64 - small.height) // 2),
        )

        cards.append(
            f'<article><strong>{html.escape(label)}</strong>'
            f'<div><img src="{png_url(image)}">'
            f'<img src="{png_url(padded)}"></div>'
            f'<small>writer={writer}, class={cid}<br>'
            f'{width}×{height} → 64×64</small></article>'
        )

    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>HWDB1.1 数据核对</title>
<style>
body{font-family:Arial,"Microsoft YaHei",sans-serif;
margin:24px;background:#f4f5f7}
main{display:grid;grid-template-columns:
repeat(auto-fit,minmax(230px,1fr));gap:12px}
article{background:white;padding:12px;
border:1px solid #ddd;border-radius:8px}
strong{font-size:32px}
img{width:100px;height:100px;object-fit:contain;margin:4px}
small{color:#555}
</style><h1>训练集随机100个样本</h1>
<p>大字是数据标签。左图为原始字图，右图为等比例缩放、
补白后的模型输入。请检查图像与标签是否对应、笔画是否被裁掉。</p>
<main>"""
    page += "".join(cards) + "</main></html>"
    (ROOT / "outputs/data_preview.html").write_text(
        page, encoding="utf-8"
    )


def main():
    for directory in ("configs", "data/processed", "outputs", "tmp"):
        (ROOT / directory).mkdir(parents=True, exist_ok=True)

    # GB2312一级汉字，共3755类；不包含字母、数字和标点。
    tags = [
        bytes((a, b))
        for a in range(0xB0, 0xD8)
        for b in range(0xA1, 0xFF)
        if a < 0xD7 or b <= 0xF9
    ]
    labels = [tag.decode("gb2312") for tag in tags]
    assert len(labels) == len(set(labels)) == 3755
    tag_to_id = {tag: i for i, tag in enumerate(tags)}
    pilot_ids = sorted(random.Random(SEED).sample(range(3755), 500))

    # 只在官方训练书写者内部划分；官方测试书写者独立保留。
    writers = list(range(1001, 1241))
    random.Random(SEED).shuffle(writers)
    splits = {
        "train": sorted(writers[:200]),
        "confusion": sorted(writers[200:220]),
        "val": sorted(writers[220:]),
        "test": list(range(1241, 1301)),
    }
    writer_to_split = {
        w: split for split, ws in splits.items() for w in ws
    }
    assert len(writer_to_split) == 300

    files = []
    for source, first, last in (
        ("train", 1001, 1240),
        ("test", 1241, 1300),
    ):
        folder = ROOT / "data/raw/hwdb1.1" / source
        found = sorted(folder.glob("*.gnt"))
        expected = {f"{w}-f.gnt" for w in range(first, last + 1)}
        if {p.name for p in found} != expected:
            raise RuntimeError(f"GNT file list mismatch: {folder}")
        files.extend(found)

    fixed_json("classes.json", {
        "encoding": "gb2312", "classes": labels,
    })
    fixed_json("writer_splits.json", {
        "seed": SEED, "splits": splits,
    })
    fixed_json("pilot_classes.json", {
        "seed": SEED,
        "global_class_ids": pilot_ids,
        "classes": [labels[i] for i in pilot_ids],
    })
    fixed_json("preprocess.json", {
        "canvas_size": 64,
        "content_max_side": 56,
        "resize": "bilinear",
        "background": 255,
        "polarity": "black_on_white",
    })

    counts, pilot_counts, excluded = Counter(), Counter(), Counter()
    class_counts = {s: Counter() for s in splits}
    pilot_set = set(pilot_ids)
    previews, seen_train = [], 0
    rng = random.Random(SEED)
    total = 0

    temporary = ROOT / "tmp/hwdb1_1_index.csv.tmp"
    destination = ROOT / "data/processed/hwdb1_1_index.csv"

    with temporary.open("w", newline="", encoding="utf-8") as output:
        table = csv.writer(output)
        table.writerow([
            "gnt_path", "pixel_offset", "width", "height",
            "class_id", "writer_id", "split",
        ])

        for n, path in enumerate(files, 1):
            writer = int(path.name.split("-")[0])
            split = writer_to_split[writer]
            relpath = path.relative_to(ROOT).as_posix()

            with path.open("rb") as f:
                for tag, width, height, offset in records(
                    f, path.stat().st_size, path.name
                ):
                    total += 1
                    if tag not in tag_to_id:
                        excluded[tag.hex()] += 1
                        continue

                    cid = tag_to_id[tag]
                    table.writerow([
                        relpath, offset, width, height,
                        cid, writer, split,
                    ])
                    counts[split] += 1
                    class_counts[split][cid] += 1

                    if cid in pilot_set:
                        pilot_counts[split] += 1

                    # 蓄水池抽样，只从训练集抽取100个核对样本。
                    if split == "train":
                        seen_train += 1
                        item = (
                            relpath, offset, width, height,
                            cid, writer, labels[cid],
                        )
                        if len(previews) < 100:
                            previews.append(item)
                        else:
                            j = rng.randrange(seen_train)
                            if j < 100:
                                previews[j] = item

            if n % 20 == 0:
                print(
                    f"Scanned {n}/300 files; "
                    f"Chinese samples={sum(counts.values())}",
                    flush=True,
                )

    if len(class_counts["train"]) != 3755:
        raise RuntimeError("Training set does not contain all 3755 classes")

    temporary.replace(destination)
    make_preview(previews)

    report = {
        "seed": SEED,
        "gnt_files": len(files),
        "all_records": total,
        "chinese_records": sum(counts.values()),
        "non_GB1_records": sum(excluded.values()),
        "writers_per_split": {
            s: len(ws) for s, ws in splits.items()
        },
        "samples_per_split": dict(counts),
        "classes_per_split": {
            s: len(c) for s, c in class_counts.items()
        },
        "pilot_classes": 500,
        "pilot_samples_per_split": dict(pilot_counts),
    }
    (ROOT / "outputs/data_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Preview:", ROOT / "outputs/data_preview.html")
    print("Index:", destination)


if __name__ == "__main__":
    main()
