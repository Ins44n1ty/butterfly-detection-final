"""Input manifests, resumable inference, and JSONL output."""

import csv
import hashlib
import io
import json
import os
import platform
import re
import tempfile
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from PIL import Image, ImageOps

from butterfly_models import Box, load_model, model_dtype, select_boxes

ROOT = Path(__file__).resolve().parent
PHOTO_COUNT = 400
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif", ".bmp")


@dataclass(frozen=True)
class Photo:
    photo_id: int
    source: str


def photo_id(value):
    if type(value) is int and value > 0:
        return value
    if isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value):
        return int(value)
    raise ValueError(f"invalid photo_id: {value!r}")


def indexed(rows):
    index = {}
    for row in rows:
        identifier = photo_id(row["photo_id"])
        if identifier in index:
            raise ValueError(f"duplicate photo_id: {identifier}")
        index[identifier] = row
    return index


def local_images(path):
    index = {}
    if not path.is_dir():
        raise ValueError(f"image directory does not exist: {path}")
    for file in sorted(path.iterdir()):
        if file.is_file() and file.suffix.lower() in IMAGE_SUFFIXES and file.stem.isdecimal():
            identifier = photo_id(file.stem)
            if identifier in index:
                raise ValueError(f"multiple files for photo_id {identifier} in {path}")
            index[identifier] = str(file.resolve())
    return index


def unique_json_object(pairs):
    row = dict(pairs)
    if len(row) != len(pairs):
        raise ValueError("duplicate JSON key")
    return row


def load_contest_dataset(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != PHOTO_COUNT:
        raise ValueError(f"expected {PHOTO_COUNT} JSONL lines, got {len(lines)}")
    photos = []
    ids = set()
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line, object_pairs_hook=unique_json_object)
            identifier, source = row["photo_id"], row["photo_url"]
            if type(identifier) is not int or identifier <= 0 or identifier in ids:
                raise ValueError(f"invalid or duplicate photo_id: {identifier!r}")
            if not isinstance(source, str):
                raise ValueError("photo_url must be a string")
            url = urlsplit(source)
            if url.scheme != "https" or not url.netloc:
                raise ValueError("photo_url must be an HTTPS URL")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}, line {number}: {error}") from error
        ids.add(identifier)
        photos.append(Photo(identifier, source))
    return photos


def team_data(root=ROOT, images_dir=None):
    manifest = json.loads((root / "data/manifest.json").read_text(encoding="utf-8"))
    gold_content = (root / "data/reference.jsonl").read_text(encoding="utf-8")
    changelog = (root / "data/review_status.csv").read_text(encoding="utf-8")
    gold_rows = indexed(json.loads(line) for line in gold_content.splitlines() if line.strip())
    reviews = indexed(csv.DictReader(io.StringIO(changelog)))
    local = local_images(images_dir) if images_dir else {}
    photos = [Photo(pid, local.get(pid, row["image_url"])) for pid, row in indexed(manifest).items()]
    if len(photos) != PHOTO_COUNT or {photo.photo_id for photo in photos} != gold_rows.keys() or gold_rows.keys() != reviews.keys():
        raise ValueError("manifest, reference and review must cover the same 400 photo_id values")
    if any(row.get("error") is not None for row in gold_rows.values()):
        raise ValueError("reference labels contain processing errors")
    gold = {identifier: read_boxes(row["boxes"]) for identifier, row in gold_rows.items()}
    return photos, gold, reviews


def read_boxes(items) -> list[Box]:
    if not isinstance(items, list):
        raise ValueError("boxes must be a list")
    boxes = []
    keys = ("left", "top", "width", "height")
    expected_keys = set(keys)
    for item in items:
        if not isinstance(item, dict) or item.keys() != expected_keys:
            raise ValueError(f"invalid bbox fields {item!r}")
        left, top, width, height = (item[key] for key in keys)
        if any(type(x) not in (float, int) or not 0 <= x <= 1 for x in (left, top, width, height)):
            raise ValueError(f"invalid normalized bbox {item}")
        right, bottom = left + width, top + height
        if not (left < right <= 1 + 1e-9 and top < bottom <= 1 + 1e-9 and left < 1 and top < 1):
            raise ValueError(f"invalid bbox area or extent {item}")
        boxes.append((left, top, min(1.0, right), min(1.0, bottom)))
    return boxes


def box_record(box):
    left, top, right, bottom = box
    return {"left": left, "top": top, "width": right - left, "height": bottom - top}


def open_image(photo, cache_dir):
    if urlsplit(photo.source).scheme not in ("https", "http"):
        with Image.open(photo.source) as image:
            return ImageOps.exif_transpose(image).convert("RGB")

    cache_key = hashlib.sha256(photo.source.encode()).hexdigest()[:20]
    cache = cache_dir / f"{photo.photo_id}-{cache_key}.img"
    if cache.is_file():
        with Image.open(cache) as image:
            return ImageOps.exif_transpose(image).convert("RGB")

    request = Request(photo.source, headers={"User-Agent": "ButterflyProject/1.0"})
    with urlopen(request, timeout=30) as response:
        contents = response.read()
    with Image.open(io.BytesIO(contents)) as image:
        converted = ImageOps.exif_transpose(image).convert("RGB")
    atomic_write(cache, contents)
    return converted


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            file.write(data)
            file.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def write_json(path, record):
    atomic_write(path, (json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def runtime_info(device, config):
    import torch

    return {
        "device": device,
        "gpu": torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
        "cuda": torch.version.cuda,
        "dtype": str(model_dtype(config, device)),
        "python": platform.python_version(),
        "versions": {name: version(name) for name in ("torch", "torchvision", "transformers", "Pillow")},
    }


def run_inference(photos, configs, device, work_dir, image_cache, attempts=2):
    """Return prediction rows for each configuration. Reuse completed photos on resume."""
    import torch
    from tqdm.auto import tqdm

    if not photos or not configs or attempts < 1:
        raise ValueError("inference requires photos, a configuration and at least one attempt")
    base = configs[0]
    runtime = runtime_info(device, base)
    key = {
        "configs": [cfg.to_dict() for cfg in configs],
        "runtime": runtime,
        "images": [(p.photo_id, p.source) for p in photos],
    }
    key_hash = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    checkpoint = work_dir / f"{base.name}-{key_hash}.json"
    errors = checkpoint.with_suffix(".errors.jsonl")
    completed = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else {}
    pending = [photo for photo in photos if completed.get(str(photo.photo_id), {}).get("boxes") is None]
    if pending:
        work_dir.mkdir(parents=True, exist_ok=True)
        model = load_model(base, device)
        threshold = max(0.0, min(cfg.threshold for cfg in configs) - 1e-6)
        try:
            model(open_image(pending[0], image_cache), threshold)
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception:
            pass  # Record a failing warmup photo during the timed run.
        for photo in tqdm(pending, desc=base.name, unit="photo"):
            identifier = str(photo.photo_id)
            first = completed.get(identifier, {}).get("attempts", 0) + 1
            for attempt in range(first, first + attempts):
                started = seconds = None
                try:
                    image = open_image(photo, image_cache)
                    started = time.perf_counter()
                    candidates = model(image, threshold)
                    boxes = {cfg.name: [box_record(box) for box in select_boxes(candidates, cfg)] for cfg in configs}
                    seconds = time.perf_counter() - started
                    error = None
                except torch.cuda.OutOfMemoryError:
                    raise
                except Exception as exc:
                    boxes = None
                    error = f"{type(exc).__name__}: {exc}"
                    if started is not None:
                        seconds = time.perf_counter() - started
                    with errors.open("a", encoding="utf-8") as log:
                        log.write(json.dumps({"photo_id": photo.photo_id, "attempt": attempt, "error": error}) + "\n")
                completed[identifier] = {"boxes": boxes, "error": error, "seconds": seconds, "attempts": attempt}
                write_json(checkpoint, completed)
                if error is None:
                    break
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    predictions = {cfg.name: [] for cfg in configs}
    for photo in photos:
        row = completed[str(photo.photo_id)]
        for name, rows in predictions.items():
            rows.append({
                "photo_id": photo.photo_id,
                "boxes": None if row["error"] else row["boxes"][name],
                "error": row["error"],
                "seconds": row["seconds"],
            })
    return predictions, runtime


def jsonl(rows):
    return "".join(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n" for row in rows).encode()


def write_submission(output, photos, rows):
    expected = {photo.photo_id for photo in photos}
    predictions = indexed(rows)
    if len(photos) != PHOTO_COUNT or len(expected) != PHOTO_COUNT or expected != predictions.keys():
        raise ValueError("submission must contain exactly the 400 distinct manifest IDs")
    failed = [identifier for identifier, row in predictions.items() if row.get("error") is not None]
    if failed:
        raise ValueError(f"{len(failed)} processing errors, photo_id examples: {failed[:10]}")
    payload = jsonl(
        {
            "photo_id": p.photo_id,
            "boxes": [box_record(box) for box in read_boxes(predictions[p.photo_id]["boxes"])],
            "error": None,
        }
        for p in photos
    )
    if len(payload) > 5 * 1024**2:
        raise ValueError("submission exceeds 5 MiB")
    atomic_write(output, payload)
