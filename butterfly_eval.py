"""Load benchmark results and compute Contest-style object metrics."""

import json
from statistics import median

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from butterfly_data import indexed, read_boxes, unique_json_object
from butterfly_models import CONTEST_PROMPTS, ModelConfig, iou


def load_saved_predictions(directory):
    predictions, configs = {}, {}
    for path in sorted(directory.glob("*.jsonl")):
        metadata = path.with_suffix(".json")
        if not metadata.is_file():
            continue
        config = ModelConfig.from_dict(json.loads(metadata.read_text(encoding="utf-8"))["model"])
        if config.backend not in CONTEST_PROMPTS:
            continue
        if config.name != path.stem:
            raise ValueError(f"Имя конфигурации не совпадает с файлом {path.name}.")
        configs[config.name] = config
        predictions[config.name] = [
            json.loads(line, object_pairs_hook=unique_json_object)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
    missing = CONTEST_PROMPTS.keys() - {config.backend for config in configs.values()}
    if missing:
        raise ValueError(f"В {directory} нужны пары .jsonl и .json для моделей: {', '.join(sorted(missing))}.")
    return predictions, configs


def ranking_key(name, overall):
    """Prefer F1, then measured processing time."""
    seconds = overall["inference_seconds"]
    return -overall["f1_iou50"], seconds if seconds is not None else float("inf"), name


def match_count(expected, predicted, min_iou):
    if not expected or not predicted:
        return 0
    links = csr_matrix([[iou(box, truth) >= min_iou for truth in expected] for box in predicted])
    matches = maximum_bipartite_matching(links, perm_type="column")
    return int((matches >= 0).sum())


def per_photo_metrics(gold, predictions):
    rows = indexed(predictions)
    if gold.keys() != rows.keys():
        raise ValueError("predictions must cover exactly the reference photo IDs")
    details = {}
    for identifier, truth in gold.items():
        row = rows[identifier]
        error = row.get("error")
        predicted = [] if error else read_boxes(row["boxes"])
        tp = match_count(truth, predicted, 0.50)
        details[identifier] = {
            "tp": tp,
            "fp": len(predicted) - tp,
            "fn": len(truth) - tp,
            "tp75": match_count(truth, predicted, 0.75),
            "truth_boxes": len(truth),
            "technical_error": bool(error),
            "seconds": row.get("seconds"),
        }
    return details


def metrics(details, ids):
    sample = [details[identifier] for identifier in ids]
    tp, fp, fn, tp75 = (sum(row[key] for row in sample) for key in ("tp", "fp", "fn", "tp75"))
    box_count = 2 * tp + fp + fn
    negative = [row for row in sample if row["truth_boxes"] == 0]
    seconds = [row["seconds"] for row in sample if row["seconds"] is not None and not row["technical_error"]]
    return {
        "photos": len(sample),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "object_errors": fp + fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1_iou50": 2 * tp / box_count if box_count else 0.0,
        "f1_iou75": 2 * tp75 / box_count if box_count else 0.0,
        "negative_photos": len(negative),
        "fp_on_negative": sum(row["fp"] for row in negative),
        "negative_photos_with_fp": sum(row["fp"] > 0 for row in negative),
        "technical_errors": sum(row["technical_error"] for row in sample),
        "inference_seconds": sum(seconds) if seconds else None,
        "median_ms": 1000 * median(seconds) if seconds else None,
    }
