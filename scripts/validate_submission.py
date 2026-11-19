import argparse
import json
import math
from pathlib import Path


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Повторяющийся ключ: {key}")
        result[key] = value
    return result


def read_jsonl(path):
    return [json.loads(line, object_pairs_hook=unique_object)
            for line in path.read_text(encoding="utf-8").splitlines()]


def validate(submission, dataset):
    if submission.stat().st_size > 5 * 1024**2:
        raise ValueError("Файл больше 5 МиБ")
    expected = read_jsonl(dataset)
    rows = read_jsonl(submission)
    if len(expected) != 400 or len(rows) != 400:
        raise ValueError("Нужны 400 строк")
    ids = [row["photo_id"] for row in rows]
    if any(type(value) is not int for value in ids) or len(set(ids)) != 400:
        raise ValueError("Некорректные или повторяющиеся photo_id")
    if set(ids) != {row["photo_id"] for row in expected}:
        raise ValueError("ID не совпадают с датасетом")
    for row in rows:
        if not set(row) <= {"photo_id", "photo_url", "boxes", "error"}:
            raise ValueError("Лишние поля")
        if row.get("error") is not None or not isinstance(row.get("boxes"), list):
            raise ValueError("Ошибка обработки или незаполненный boxes")
        for box in row["boxes"]:
            if set(box) != {"left", "top", "width", "height"}:
                raise ValueError("Некорректные поля рамки")
            if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
                   for v in box.values()):
                raise ValueError("Некорректные координаты")
            if (box["width"] <= 0 or box["height"] <= 0
                    or box["left"] + box["width"] > 1 + 1e-9
                    or box["top"] + box["height"] > 1 + 1e-9):
                raise ValueError("Некорректный размер рамки")
    return rows


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--dataset", type=Path, default=root / "data/contest/dataset.jsonl")
    args = parser.parse_args()
    rows = validate(args.submission, args.dataset)
    print(f"Проверены {len(rows)} фото, {sum(len(r['boxes']) for r in rows)} рамки")


if __name__ == "__main__":
    main()
