#!/usr/bin/env python3
"""Generate a validated JSONL file for Yandex Contest."""

import argparse
import json
from pathlib import Path

from butterfly_data import ROOT, load_contest_dataset, run_inference, write_submission
from butterfly_models import ModelConfig, resolve_device


def main():
    parser = argparse.ArgumentParser(
        description="Сформировать JSONL для Контеста на 400 экзаменационных фотографиях.",
    )
    parser.add_argument("dataset", type=Path, help="официальный dataset.jsonl с полями photo_id и photo_url")
    parser.add_argument("-o", "--output", type=Path, default=Path("submission.jsonl"))
    parser.add_argument(
        "--config", type=Path, default=ROOT / "model_results/winner_config.json",
        help="настройки модели, сохранённые ноутбуком",
    )
    parser.add_argument("--device", default="cuda", help="устройство PyTorch, например cuda:1")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "data/contest", help="папка для кеша и журнала")
    parser.add_argument("--attempts", type=int, default=2, help="попыток обработки на фото за запуск")
    args = parser.parse_args()
    try:
        if args.output.suffix != ".jsonl" or args.output.resolve() == args.dataset.resolve():
            raise ValueError("Укажите отдельный выходной файл с расширением .jsonl.")
        photos = load_contest_dataset(args.dataset)
        if not args.config.is_file():
            raise ValueError(f"запустите сравнение в ноутбуке для получения {args.config}")
        config = ModelConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
        if not config.includes_moths:
            raise ValueError("Нужны настройки победителя, который искал бабочек и молей.")
        device = resolve_device(args.device)
        predictions, _ = run_inference(
            photos, (config,), device, args.work_dir, args.work_dir / "images", args.attempts,
        )
        write_submission(args.output, photos, predictions[config.name])
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\nЖурнал и сохранённый прогресс: {args.work_dir.resolve()}\n")
    print(f"{len(photos)} предсказаний → {args.output.resolve()}")


if __name__ == "__main__":
    main()
