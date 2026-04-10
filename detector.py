"""
Cascade detector — public API used by the Processor.

Decision logic:
    DINO conf >= threshold and box found -> accept immediately (source: dino)
    Otherwise -> escalate to LLM (source: llm)

When LLM is unavailable, use a conservative DINO fallback:
    - if DINO found a box and conf >= LOW_THRESHOLD -> accept (source: heuristic)
    - else -> flag

Thresholds are set relative to the user-supplied --threshold flag:
    LOW_THRESHOLD  = threshold - 0.10   (heuristic fallback gate)
    LLM threshold  = max(0.15, threshold * 0.70)

This means at the default threshold of 0.35:
    >= 0.35 -> DINO accepts directly
    < 0.35  -> LLM verifies (if available)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

from detector_dino import DetectionResult, DinoDetector
from detector_llm import LLMDetector


class SignatureDetector:
    """
    Cascade: Grounding DINO -> (uncertain only) -> Ollama LLM.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.35,
        device: str | None = None,
        model: str = "moondream:latest",
        ollama_url: str = "http://localhost:11434",
        llm_workers: int = 2,
    ):
        self.threshold = confidence_threshold
        self.low_threshold = max(0.0, confidence_threshold - 0.10)
        self.llm_threshold = max(0.15, confidence_threshold * 0.70)
        self.llm_workers = max(1, int(llm_workers))

        print(
            f"\n[Cascade] Thresholds - accept: >= {self.threshold:.2f} | "
            f"escalate: < {self.threshold:.2f} | "
            f"llm_threshold: {self.llm_threshold:.2f} | "
            f"fallback_gate: {self.low_threshold:.2f}"
        )

        self.dino = DinoDetector(device=device)
        self.llm = LLMDetector(model=model, ollama_url=ollama_url)

    def detect(self, pil_image: Image.Image) -> DetectionResult:
        """Single-image compatibility wrapper around batched detection."""
        return self.detect_batch([pil_image])[0]

    def detect_batch(self, pil_images: list[Image.Image]) -> list[DetectionResult]:
        """
        Batched cascade detection.
        1) DINO runs as a batch on GPU/CPU.
        2) Only uncertain results are escalated to LLM.
        3) LLM escalations can be parallelized.
        """
        if not pil_images:
            return []

        dino_results = self.dino.detect_batch(pil_images)
        final_results: list[DetectionResult] = [
            DetectionResult(
                found=r.found,
                box=r.box,
                confidence=r.confidence,
                reason=r.reason,
                source=r.source,
                meta=r.meta or {},
            )
            for r in dino_results
        ]

        uncertain_indices: list[int] = []
        for idx, dino_result in enumerate(dino_results):
            conf = dino_result.confidence

            if dino_result.found and dino_result.box is not None and conf >= self.threshold:
                continue

            if not self.llm.available:
                # Conservative fallback when LLM is unavailable.
                if dino_result.found and conf >= self.low_threshold and dino_result.box is not None:
                    final_results[idx] = DetectionResult(
                        found=True,
                        box=dino_result.box,
                        confidence=conf,
                        reason=f"LLM unavailable; accepted DINO fallback at conf {conf:.3f}",
                        source="heuristic",
                        meta=dino_result.meta or {},
                    )
                else:
                    final_results[idx] = DetectionResult(
                        found=False,
                        box=None,
                        confidence=conf,
                        reason=f"LLM unavailable and DINO conf {conf:.3f} below fallback gate",
                        source="dino",
                        meta=dino_result.meta or {},
                    )
                continue

            uncertain_indices.append(idx)

        if not uncertain_indices:
            return final_results

        if self.llm_workers == 1 or len(uncertain_indices) == 1:
            for idx in uncertain_indices:
                final_results[idx] = self.llm.verify(
                    pil_images[idx],
                    dino_results[idx],
                    self.llm_threshold,
                )
            return final_results

        max_workers = min(self.llm_workers, len(uncertain_indices))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {
                pool.submit(self.llm.verify, pil_images[idx], dino_results[idx], self.llm_threshold): idx
                for idx in uncertain_indices
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    final_results[idx] = future.result()
                except Exception as exc:
                    conf = dino_results[idx].confidence
                    final_results[idx] = DetectionResult(
                        found=False,
                        box=None,
                        confidence=conf,
                        reason=f"LLM exception ({type(exc).__name__}) - flagged",
                        source="llm",
                        meta={"llm_error": "exception"},
                    )

        return final_results
