import argparse
import csv
import math
from pathlib import Path

from butterfly_data import ROOT, team_data
from butterfly_eval import load_saved_predictions, metrics, per_photo_metrics, ranking_key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    photos, gold, reviews = team_data(gold_path=args.gold)
    predictions, configs = load_saved_predictions(ROOT / "model_results")
    slices = {
        "все фото": list(gold),
        "принятые на ревью": [p for p in gold if reviews[p]["review_resolution"] == "ACCEPTED"],
        "без рамок": [p for p, boxes in gold.items() if not boxes],
        "одна рамка": [p for p, boxes in gold.items() if len(boxes) == 1],
        "несколько рамок": [p for p, boxes in gold.items() if len(boxes) > 1],
        "мелкие объекты": [p for p, boxes in gold.items()
                           if any((x2-x1)*(y2-y1) <= 0.05 for x1, y1, x2, y2 in boxes)],
    }
    report = []
    for name, rows in predictions.items():
        details = per_photo_metrics(gold, rows)
        for label, ids in slices.items():
            report.append({"model": name, "target": configs[name].target, "slice": label,
                           **metrics(details, ids)})
    output = ROOT / "model_results/leaderboard.csv"
    if args.check:
        with output.open(encoding="utf-8", newline="") as file:
            saved = {(r["model"], r["slice"]): r for r in csv.DictReader(file)}
        assert set(saved) == {(r["model"], r["slice"]) for r in report}
        for row in report:
            original = saved[row["model"], row["slice"]]
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    assert math.isclose(value, float(original[key]), rel_tol=1e-9, abs_tol=1e-9), (row["model"], row["slice"], key)
        print("Все метрики совпадают с сохранённой таблицей")
    else:
        with output.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(report[0]))
            writer.writeheader()
            writer.writerows(report)
    overall = {r["model"]: r for r in report if r["slice"] == "все фото"}
    for name in sorted(overall, key=lambda n: ranking_key(n, overall[n])):
        row = overall[name]
        print(f"{name}: F1@0.50={row['f1_iou50']:.4f}, F1@0.75={row['f1_iou75']:.4f}")


if __name__ == "__main__":
    main()

