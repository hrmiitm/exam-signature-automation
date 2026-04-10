"""
Grounding DINO detector (Tier 2a).

Fast zero-shot object detector. Returns a DetectionResult with a
confidence score. High-confidence results are accepted immediately;
uncertain results are escalated to the Ollama LLM (Tier 2b).

Model: IDEA-Research/grounding-dino-tiny  (~680 MB, cached on first run)
"""

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from typing import Any, Optional

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_ID       = "IDEA-Research/grounding-dino-tiny"
PROMPT         = "handwritten signature."
BOX_THRESHOLD  = 0.20   # internal DINO box threshold (keep low — we filter by confidence ourselves)
TEXT_THRESHOLD = 0.15


@dataclass
class DetectionResult:
    found:      bool
    box:        Optional[tuple[int, int, int, int]]   # (x1, y1, x2, y2) absolute pixels
    confidence: float
    reason:     str
    source:     str   # "dino" | "llm" | "heuristic"
    meta:       dict[str, Any] | None = None


class DinoDetector:
    """Wraps Grounding DINO. Load once, call .detect() many times."""

    def __init__(self, device: str | None = None):
        self.requested_device = device or _best_device()
        self.device = self.requested_device
        print(f"[DINO] Loading {MODEL_ID} on {self.requested_device} ...")
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model     = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID)
        self._move_model_to_device_with_fallback()
        self.model.eval()
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
        sig = inspect.signature(self.processor.post_process_grounded_object_detection)
        self._uses_box_threshold = "box_threshold" in sig.parameters
        print(f"[DINO] Ready on {self.device}.")

    def _move_model_to_device_with_fallback(self) -> None:
        try:
            if self.requested_device == "cuda" and torch.cuda.is_available():
                try:
                    free_bytes, total_bytes = torch.cuda.mem_get_info()
                    free_mb = free_bytes / (1024 * 1024)
                    total_mb = total_bytes / (1024 * 1024)
                    # If free memory is critically low, avoid immediate OOM and fall back.
                    if free_mb < 512:
                        print(
                            f"[DINO] CUDA free memory too low ({free_mb:.1f} MiB / {total_mb:.1f} MiB). "
                            "Falling back to CPU."
                        )
                        self.device = "cpu"
                        self.model = self.model.to(self.device)
                        return
                except Exception:
                    # If mem query fails, continue with normal move attempt.
                    pass

            self.model = self.model.to(self.requested_device)
            self.device = self.requested_device
        except RuntimeError as exc:
            msg = str(exc).lower()
            is_cuda_oom = "outofmemory" in msg or ("out of memory" in msg and "cuda" in msg)
            if self.requested_device == "cuda" and is_cuda_oom:
                print("[DINO] CUDA OOM while loading model; falling back to CPU.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.device = "cpu"
                self.model = self.model.to(self.device)
                return
            raise

    def detect(self, pil_image: Image.Image) -> DetectionResult:
        return self.detect_batch([pil_image])[0]

    def detect_batch(self, pil_images: list[Image.Image]) -> list[DetectionResult]:
        """Run batched DINO inference for higher throughput, especially on GPU."""
        if not pil_images:
            return []

        return self._detect_batch_with_backoff(pil_images)

    def _detect_batch_with_backoff(self, pil_images: list[Image.Image]) -> list[DetectionResult]:
        """Retry oversized GPU batches by splitting them to avoid OOM crashes."""
        if not pil_images:
            return []

        images_rgb = [img.convert("RGB") for img in pil_images]
        target_sizes = [(img.height, img.width) for img in images_rgb]
        text_batch = [PROMPT] * len(images_rgb)

        try:
            inputs = self.processor(
                images=images_rgb,
                text=text_batch,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
        except RuntimeError as exc:
            msg = str(exc).lower()
            is_oom = "out of memory" in msg or "cuda" in msg and "memory" in msg
            if is_oom:
                if self.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if len(pil_images) == 1:
                    return [
                        DetectionResult(
                            found=False,
                            box=None,
                            confidence=0.0,
                            reason="DINO OOM on single image",
                            source="error",
                            meta={"error": "dino_oom"},
                        )
                    ]

                mid = len(pil_images) // 2
                left = self._detect_batch_with_backoff(pil_images[:mid])
                right = self._detect_batch_with_backoff(pil_images[mid:])
                return left + right

            raise

        post_kwargs = {
            "text_threshold": TEXT_THRESHOLD,
            "target_sizes": target_sizes,
        }
        if self._uses_box_threshold:
            post_kwargs["box_threshold"] = BOX_THRESHOLD
        else:
            post_kwargs["threshold"] = BOX_THRESHOLD

        results_list = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            **post_kwargs,
        )

        detections: list[DetectionResult] = []
        for image_rgb, results in zip(images_rgb, results_list):
            detections.append(self._parse_single_result(results, image_rgb.size))

        return detections

    def _parse_single_result(self, results: dict[str, Any], size: tuple[int, int]) -> DetectionResult:
        w, h = size
        boxes = results["boxes"]
        scores = results["scores"]

        if len(scores) == 0:
            return DetectionResult(
                found=False,
                box=None,
                confidence=0.0,
                reason="no detections from DINO",
                source="dino",
                meta={},
            )

        best_idx = int(torch.argmax(scores))
        best_score = float(scores[best_idx])

        x1, y1, x2, y2 = boxes[best_idx].tolist()
        box = (
            max(0, int(x1)),
            max(0, int(y1)),
            min(w, int(x2)),
            min(h, int(y2)),
        )

        return DetectionResult(
            found=True,
            box=box,
            confidence=best_score,
            reason=f"DINO detected with confidence {best_score:.3f}",
            source="dino",
            meta={},
        )


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
