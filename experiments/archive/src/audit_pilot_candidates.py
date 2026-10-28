"""CPU-only descriptive audit of saved, calibrated pilot predictions.

No model fitting, image loading, test-set access, p-values, or confidence intervals.
Leave-one-writer-out ranges are sensitivity summaries, NOT confidence intervals.
"""

import argparse
import csv
import hashlib
import itertools
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path("/workspace/shouxieti")
METHODS = ("aug_ce", "kd", "margin_hard", "margin_delta")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inside_root(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT.resolve()), f"Path is outside {ROOT}: {path}")
    return path


def load_predictions(path, index):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in
                ("predictions", "targets", "conditions", "classes", "global_class_ids")}
    predictions, targets = data["predictions"], data["targets"]
    conditions = data["conditions"].tolist()
    require(predictions.shape == (len(conditions), len(index)), f"Bad shape: {path}")
    require(targets.shape == (len(index),), f"Bad target shape: {path}")
    require(len(conditions) == len(set(conditions)) and conditions[0] == "clean", "Invalid conditions")
    require(np.issubdtype(predictions.dtype, np.integer), "Predictions must be integer class IDs")
    require(np.issubdtype(targets.dtype, np.integer), "Targets must be integer class IDs")
    require(np.all((predictions >= 0) & (predictions < len(data["classes"]))), "Invalid prediction class")
    require(np.array_equal(targets, [int(row["local_class_id"]) for row in index]), "Sample order mismatch")
    require(np.all((targets >= 0) & (targets < len(data["classes"]))), "Invalid target class")
    require(len(data["classes"]) == len(data["global_class_ids"]), "Class mapping size mismatch")
    require(np.array_equal(data["global_class_ids"][targets], [int(r["global_class_id"]) for r in index]),
            "Global label mismatch")
    require(np.array_equal(data["classes"][targets], [r["character"] for r in index]), "Character mismatch")
    return data


def paired_description(a, b, writers):
    """a,b are K x N correctness arrays. Positive differences favor b."""
    require(a.shape == b.shape and a.ndim == 2, "Unpaired correctness arrays")
    unique = np.unique(writers)
    require(len(unique) >= 2 and len(writers) == a.shape[1], "Need at least two writers")
    writer_rows, leave_one_out = [], []
    for writer in unique:
        selected = writers == writer
        aa, bb = a[:, selected], b[:, selected]
        gains = int((~aa & bb).sum())
        losses = int((aa & ~bb).sum())
        writer_rows.append({"writer_id": int(writer), "source_images": int(selected.sum()),
                            "image_condition_evaluations": int(aa.size),
                            "a_accuracy_pct": 100 * float(aa.mean()), "b_accuracy_pct": 100 * float(bb.mean()),
                            "gain_evaluations": gains, "loss_evaluations": losses,
                            "net_gain_evaluations": gains - losses,
                            "delta_pp": 100 * (gains - losses) / aa.size})
        leave_one_out.append(100 * float((b[:, ~selected].astype(np.int8)
                                        - a[:, ~selected].astype(np.int8)).mean()))
    gains, losses = int((~a & b).sum()), int((a & ~b).sum())
    return {"source_images": a.shape[1], "conditions": a.shape[0],
            "image_condition_evaluations": int(a.size), "writers": len(unique),
            "a_accuracy_pct": 100 * float(a.mean()), "b_accuracy_pct": 100 * float(b.mean()),
            "delta_pp": 100 * (gains - losses) / a.size,
            "gain_evaluations": gains, "loss_evaluations": losses, "net_gain_evaluations": gains - losses,
            "writers_b_better": sum(r["net_gain_evaluations"] > 0 for r in writer_rows),
            "writers_tied": sum(r["net_gain_evaluations"] == 0 for r in writer_rows),
            "writers_b_worse": sum(r["net_gain_evaluations"] < 0 for r in writer_rows),
            "leave_one_writer_out_min_pp": min(leave_one_out),
            "leave_one_writer_out_max_pp": max(leave_one_out)}, writer_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-run", type=Path, required=True)
    options = parser.parse_args()
    run = inside_root(options.candidates_run)
    config = read_json(run / "config.json")
    controls, transfer = inside_root(config["controls_run"]), inside_root(config["transfer_run"])
    require(read_json(run / "result.json")["completed"], "Candidate run incomplete")
    require(read_json(transfer / "result.json")["completed"], "Transfer run incomplete")
    require(read_json(transfer / "config.json")["controls_config"] == read_json(controls / "config.json"),
            "Transfer / controls provenance mismatch")
    require(config["data_sha256"] == read_json(controls / "config.json")["data_sha256"], "Data fingerprints differ")
    index = read_csv(run / "val_sample_index.csv")
    require([int(r["row"]) for r in index] == list(range(len(index))), "Invalid index order")
    require(len({(r["gnt_path"], r["pixel_offset"]) for r in index}) == len(index), "Duplicate samples")
    index_hash = sha256(run / "val_sample_index.csv")
    for directory in (controls, transfer):
        require(sha256(directory / "val_sample_index.csv") == index_hash, "Sample index files differ")
    writers = np.asarray([int(row["writer_id"]) for row in index])
    expected_writers = config["data_config"]["writer_splits.json"]["splits"]["val"]
    require(set(writers.tolist()) == set(expected_writers), "Validation writers differ from protocol")
    suites, input_hashes = {}, {}
    for suite in ("seen", "unseen"):
        methods = {}
        for method in METHODS:
            path = ((controls / "aug_ce/predictions_calibrated_bn.npz" if suite == "seen"
                     else transfer / "predictions_aug_ce.npz") if method == "aug_ce"
                    else run / method / suite / "predictions_calibrated_bn.npz")
            methods[method] = load_predictions(path, index)
            input_hashes[str(path)] = sha256(path)
        reference = methods["aug_ce"]
        require(len(reference["conditions"]) == (7 if suite == "seen" else 9), "Unexpected protocol size")
        protocol_names = [c["name"] for c in config[f"{suite}_protocol"]["conditions"]]
        require(reference["conditions"].tolist() == protocol_names, "Evaluation protocol mismatch")
        for method, data in methods.items():
            for key in ("conditions", "targets", "classes", "global_class_ids"):
                require(np.array_equal(data[key], reference[key]), f"Unpaired {method} {key}")
        # Reproduce every reported condition and the mean from the saved predictions.
        table = read_csv(run / f"comparison_{suite}.csv")
        require([r["condition"] for r in table] == protocol_names + [f"mean_of_{len(protocol_names)-1}_{suite}"],
                "Comparison table row mismatch")
        for method, data in methods.items():
            accuracy = (data["predictions"] == data["targets"][None, :]).mean(1) * 100
            reported = [float(r[f"{method}_pct"]) for r in table]
            np.testing.assert_allclose(reported, np.r_[accuracy, accuracy[1:].mean()], rtol=0, atol=1e-8)
        suites[suite] = methods
    for method in METHODS:
        require(np.array_equal(suites["seen"][method]["predictions"][0],
                               suites["unseen"][method]["predictions"][0]), "Clean predictions differ across suites")
    summaries, per_writer = [], []
    for suite, methods in suites.items():
        names = methods["aug_ce"]["conditions"].tolist()
        endpoints = [(name, [i]) for i, name in enumerate(names) if suite == "seen" or name != "clean"]
        endpoints.append((f"mean_of_{len(names)-1}_{suite}", list(range(1, len(names)))))
        for a_name, b_name in itertools.combinations(METHODS, 2):
            for endpoint, positions in endpoints:
                a = methods[a_name]["predictions"][positions] == methods[a_name]["targets"][None, :]
                b = methods[b_name]["predictions"][positions] == methods[b_name]["targets"][None, :]
                stats, detailed = paired_description(a, b, writers)
                prefix = {"suite": suite, "endpoint": endpoint, "a": a_name, "b": b_name}
                summaries.append({**prefix, **stats})
                per_writer.extend({**prefix, **row} for row in detailed)
    # Export low-resolution cases where the methods differ in correctness.
    methods = suites["unseen"]
    position = methods["aug_ce"]["conditions"].tolist().index("lowres_20")
    correct = np.stack([methods[m]["predictions"][position] == methods[m]["targets"] for m in METHODS])
    changed = np.flatnonzero(correct.any(0) & ~correct.all(0))
    examples = []
    for i in changed:
        row = dict(index[int(i)])
        for m in METHODS:
            prediction = int(methods[m]["predictions"][position, i])
            row[f"{m}_prediction"] = str(methods[m]["classes"][prediction])
            row[f"{m}_correct"] = bool(prediction == methods[m]["targets"][i])
        examples.append(row)
    out = ROOT / "outputs" / f"candidate_audit_{datetime.now():%Y%m%d_%H%M%S_%f}"
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "paired_summary.csv", summaries)
    write_csv(out / "by_writer.csv", per_writer)
    fields = list(index[0]) + [f"{m}_{suffix}" for m in METHODS for suffix in ("prediction", "correct")]
    write_csv(out / "lowres20_disagreements.csv", examples, fields)
    shutil.copyfile(Path(__file__), out / Path(__file__).name)
    shutil.copyfile(run / "val_sample_index.csv", out / "val_sample_index.csv")
    result = {"completed": True, "candidates_run": str(run), "source_images": len(index),
              "validation_writers": len(np.unique(writers)), "numpy_version": np.__version__,
              "input_sha256": input_hashes, "sample_index_sha256": index_hash,
              "source_sha256": sha256(Path(__file__)), "reported_metrics_reproduced": True,
              "analysis": "descriptive paired counts and leave-one-writer-out sensitivity; no exclusions",
              "boundaries": "one training seed; reused validation; synthetic conditions share source images; no inferential tests; leave-one-out range is NOT a confidence interval",
              "test_evaluated": False, "training_performed": False}
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PASS: sample/class/protocol alignment; all reported accuracies reproduced. {len(index)} images, {len(np.unique(writers))} writers.")
    for row in summaries:
        if row["endpoint"] in ("clean", "mean_of_6_seen", "mean_of_8_unseen", "lowres_20"):
            print(f"{row['endpoint']:17} {row['b']:12} vs {row['a']:12} "
                  f"delta={row['delta_pp']:+.4f}pp gains/losses={row['gain_evaluations']}/{row['loss_evaluations']} "
                  f"writers(+/0/-)={row['writers_b_better']}/{row['writers_tied']}/{row['writers_b_worse']} "
                  f"leave-one-writer-out=[{row['leave_one_writer_out_min_pp']:+.4f},{row['leave_one_writer_out_max_pp']:+.4f}]pp")
    print("Counts for mean endpoints are image-condition evaluations, not independent images.")
    print("Leave-one-writer-out ranges are sensitivity summaries, NOT confidence intervals.")
    print(f"DONE: {out}")


if __name__ == "__main__":
    main()
