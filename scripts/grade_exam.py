import json
from collections import defaultdict
from pathlib import Path

from butterfly_data import read_boxes
from butterfly_eval import match_count


def main():
    root = Path(__file__).resolve().parent.parent
    config = json.loads((root / "annotation/admission.json").read_text())
    gold = {x["photo_id"]: x for x in json.loads((root / "annotation/exam_gold.json").read_text())}
    answers = defaultdict(dict)
    for line in (root / "data/team/exam_answers.jsonl").read_text().splitlines():
        row = json.loads(line)
        answers[row["worker_id"]][row["photo_id"]] = row
    admitted = []
    for worker, rows in answers.items():
        correct = {}
        for pid, target in gold.items():
            row = rows.get(pid, {})
            try:
                predicted = read_boxes(row.get("boxes", []))
            except ValueError:
                correct[pid] = False
                continue
            expected = read_boxes(target["boxes"])
            correct[pid] = (row.get("status") == target["status"]
                            and len(predicted) == len(expected)
                            and match_count(expected, predicted, config["minimum_iou"]) == len(expected))
        if (rows.keys() == gold.keys()
                and sum(correct.values()) >= config["minimum_correct"]
                and all(correct[pid] for pid in config["required_correct_photo_ids"])):
            admitted.append(worker)
    assert len(admitted) == config["admitted_workers"]
    print(f"Проверены {len(answers)} экзаменов, допуск получили {len(admitted)} исполнителя")


if __name__ == "__main__":
    main()
