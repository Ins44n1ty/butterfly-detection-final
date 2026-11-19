"""Shared inference for the benchmark and the Contest."""

import json
import math
import re
from dataclasses import asdict, dataclass, replace

from PIL import Image

Box = tuple[float, float, float, float]  # normalized x1, y1, x2, y2
Candidate = tuple[Box, float]
CONTEST_TARGET = "adult_butterflies_with_moths"

QWEN_PROMPT = (
    "Detect every adult butterfly or moth in this image. Include partially visible "
    "individuals. Draw one tight box around the visible parts of each individual, "
    "including visible antennae and legs. Exclude caterpillars, pupae, other insects "
    "and shadows. Return only a JSON array. Each item must be an object like "
    '{"bbox_2d":[x_min,y_min,x_max,y_max]} with coordinates normalized to 0–1000. '
    "Return [] if there are no adult butterflies or moths."
)
CONTEST_PROMPTS = {
    "dino": ("butterfly. moth.",),
    "owl": ("a photo of a butterfly", "a photo of a moth"),
    "qwen_torch": (QWEN_PROMPT,),
}


@dataclass(frozen=True)
class ModelConfig:
    name: str
    backend: str
    repo: str
    revision: str
    prompts: tuple[str, ...]
    threshold: float = 0.0
    nms_iou: float = 0.70
    qwen_max_side: int = 640
    max_new_tokens: int = 1024
    target: str = CONTEST_TARGET

    def to_dict(self):
        return {**asdict(self), "prompts": list(self.prompts)}

    @classmethod
    def from_dict(cls, data):
        prompts = tuple(data["prompts"])
        # Earlier versions saved the prompt without a separate target field.
        legacy_target = CONTEST_TARGET if prompts == CONTEST_PROMPTS.get(data["backend"]) else "unknown"
        return cls(**{**data, "prompts": prompts, "target": data.get("target", legacy_target)})

    @property
    def includes_moths(self):
        return self.backend in CONTEST_PROMPTS and self.target == CONTEST_TARGET

    def with_moths(self):
        return replace(
            self, name=f"{self.name}_with_moths", prompts=CONTEST_PROMPTS[self.backend], target=CONTEST_TARGET,
        )


def benchmark_configs():
    dino = ModelConfig(
        "grounding_dino_tiny", "dino", "IDEA-Research/grounding-dino-tiny",
        "a2bb814dd30d776dcf7e30523b00659f4f141c71", CONTEST_PROMPTS["dino"],
        threshold=0.30,
    )
    owl = ModelConfig(
        "owlv2_base", "owl", "google/owlv2-base-patch16-ensemble",
        "cfd3195ba4ea9592eec887ded089f4c08eff231d",
        CONTEST_PROMPTS["owl"], threshold=0.15,
    )
    qwen = ModelConfig(
        "qwen3_vl_2b", "qwen_torch", "Qwen/Qwen3-VL-2B-Instruct",
        "89644892e4d85e24eaac8bacfd4f463576704203", CONTEST_PROMPTS["qwen_torch"],
    )
    return {
        "grounding_dino": (dino, replace(dino, name="grounding_dino_tiny_020", threshold=0.20)),
        "owlv2": (owl,),
        "qwen": (qwen,),
    }


def normalized_box(coords, width=1, height=1):
    if not isinstance(coords, (list, tuple)) or len(coords) != 4:
        raise ValueError("expected four bbox coordinates")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in coords):
        raise ValueError("bbox coordinates must be finite numbers")
    x1, y1, x2, y2 = (
        min(1.0, max(0.0, value / scale))
        for value, scale in zip(coords, (width, height, width, height))
    )
    if x1 >= x2 or y1 >= y2:
        raise ValueError("bbox has zero or negative area after clipping")
    return x1, y1, x2, y2


def iou(a: Box, b: Box) -> float:
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return overlap / (area_a + area_b - overlap)


def select_boxes(candidates: list[Candidate], config: ModelConfig) -> list[Box]:
    selected = []
    for box, score in sorted(candidates, key=lambda item: item[1], reverse=True):
        if score >= config.threshold and all(iou(box, other) <= config.nms_iou for other in selected):
            selected.append(box)
    return selected


def parse_qwen(text: str) -> list[Candidate]:
    try:
        text = text.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        rows = json.loads(fenced[1] if fenced else text)
        if not isinstance(rows, list):
            raise ValueError("expected a JSON array")
        return [(normalized_box(row["bbox_2d"], 1000, 1000), 1.0) for row in rows]
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"invalid Qwen output ({exc}): {text[:2000]!r}") from exc


def resolve_device(requested):
    import torch

    device = torch.device(requested)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Укажите cuda, cuda:N или cpu.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна. Проверьте драйвер NVIDIA и сборку PyTorch в окружении ядра.")
    if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(f"GPU {device} не виден PyTorch.")
    return str(device)


def model_dtype(config, device):
    import torch

    if config.backend == "qwen_torch" and device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    return torch.float32


class Detector:
    def __init__(self, config: ModelConfig, device: str):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.config = config
        self.device = device
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            config.repo, revision=config.revision, use_fast=True,
        )
        kernel_options = {"disable_custom_kernels": True} if config.backend == "dino" else {}
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            config.repo, revision=config.revision, use_safetensors=True, **kernel_options,
        ).eval().to(device)

    def __call__(self, image, threshold):
        config = self.config
        text = config.prompts[0] if config.backend == "dino" else [list(config.prompts)]
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        width, height = image.size
        with self.torch.inference_mode():
            output = self.model(**inputs)
            if config.backend == "dino":
                result = self.processor.post_process_grounded_object_detection(
                    output, inputs.input_ids, threshold=threshold, text_threshold=0.25,
                    target_sizes=[(height, width)],
                )[0]
            else:
                result = self.processor.post_process_object_detection(
                    output, threshold=threshold, target_sizes=[(height, width)],
                )[0]
        candidates = []
        for coords, score in zip(result["boxes"].tolist(), result["scores"].tolist()):
            try:
                box = normalized_box(coords, width, height)
            except ValueError:
                # A detector proposal can lie wholly outside the frame.
                continue
            candidates.append((box, score))
        return candidates


class Qwen:
    def __init__(self, config: ModelConfig, device: str):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.config = config
        self.device = device
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            config.repo, revision=config.revision, use_fast=True,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.repo, revision=config.revision, dtype=model_dtype(config, device), use_safetensors=True,
        ).eval().to(device)

    def __call__(self, image, threshold):
        config = self.config
        image = image.copy()
        image.thumbnail((config.qwen_max_side, config.qwen_max_side), Image.Resampling.LANCZOS)
        if min(image.size) < 32:
            image = image.resize(tuple(max(32, side) for side in image.size), Image.Resampling.BICUBIC)
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": config.prompts[0]}],
        }]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=config.max_new_tokens, do_sample=False,
            )
        text = self.processor.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return parse_qwen(text)


def load_model(config, device):
    if config.backend in ("dino", "owl"):
        return Detector(config, device)
    if config.backend == "qwen_torch":
        return Qwen(config, device)
    raise ValueError(f"unknown backend: {config.backend}")
