"""
Ollama LLM vision detector (Tier 2b).

Only called for images where DINO returned an uncertain confidence score.
Given the image AND DINO's candidate box, the LLM makes the final call:
  - Is there really a signature here?
  - If yes, refine the bounding box.

Returns a DetectionResult with source="llm".
"""

from __future__ import annotations

import base64
import io
import json
import re
from typing import Any, Optional

import requests
from PIL import Image

from detector_dino import DetectionResult


DEFAULT_MODEL   = "moondream:latest"
OLLAMA_BASE_URL = "http://localhost:11434"
REQUEST_TIMEOUT = 120


def _make_prompt(has_dino_box: bool) -> str:
    """Build the prompt. If DINO found a candidate box we tell the LLM about it."""
    context = (
        "Grounding DINO has already found a candidate region that might be a signature, "
        "but it is not confident. Please examine the full image carefully."
        if has_dino_box else
        "A previous detector found nothing, but please double-check the full image."
    )
    return f"""You are a document analysis expert specialising in signature detection.

{context}

A handwritten signature is:
- Cursive or stylised handwriting (a person's name or initials)
- Usually at the bottom of a document
- Visually distinct from printed or typed text
- May have flourishes, underlines, or loops

Respond ONLY with a valid JSON object — no markdown, no extra text:

If a signature IS present:
{{
  "found": true,
  "confidence": <float 0.0–1.0>,
  "box": {{
    "x1_pct": <left %>,
    "y1_pct": <top %>,
    "x2_pct": <right %>,
    "y2_pct": <bottom %>
  }},
  "reason": "<one sentence>"
}}

If NO signature is present:
{{
  "found": false,
  "confidence": 0.0,
  "box": null,
  "reason": "<one sentence>"
}}"""


class LLMDetector:
    """Ollama vision LLM. Called only for uncertain DINO results."""

    def __init__(
        self,
        model:      str = DEFAULT_MODEL,
        ollama_url: str = OLLAMA_BASE_URL,
    ):
        self.model   = model
        self.base_url = ollama_url
        self.api_url = f"{ollama_url}/api/generate"
        print(f"[LLM ] Using Ollama model : {model}")
        self.available = self._check_ollama(ollama_url)

    # ── Public ────────────────────────────────────────────────────────────────

    def verify(
        self,
        pil_image:        Image.Image,
        dino_result:      DetectionResult,
        confidence_threshold: float,
    ) -> DetectionResult:
        """
        Ask the LLM to verify an uncertain DINO detection.
        Returns a new DetectionResult with source="llm".
        """
        image_b64 = _encode_image(pil_image)
        w, h      = pil_image.size
        prompt    = _make_prompt(has_dino_box=dino_result.box is not None)

        if not self.available:
            return DetectionResult(
                found=False,
                box=None,
                confidence=dino_result.confidence,
                reason="LLM unavailable; DINO was uncertain - flagged",
                source="llm",
                meta={"llm_error": "unavailable"},
            )

        try:
            payload = self._call_ollama(image_b64, prompt)
        except requests.exceptions.ConnectionError:
            # Ollama unreachable — fall back to DINO's uncertain result
            return DetectionResult(
                found=False, box=None, confidence=dino_result.confidence,
                reason="LLM unreachable; DINO was uncertain — flagged",
                source="llm",
                meta={"llm_error": "connection_error"},
            )
        except requests.exceptions.Timeout:
            return DetectionResult(
                found=False, box=None, confidence=dino_result.confidence,
                reason="LLM timed out; DINO was uncertain — flagged",
                source="llm",
                meta={"llm_error": "timeout"},
            )

        raw_text = str(payload.get("response", ""))
        llm_meta = {
            "llm_model": self.model,
            "llm_prompt_tokens": int(payload.get("prompt_eval_count", 0) or 0),
            "llm_completion_tokens": int(payload.get("eval_count", 0) or 0),
            "llm_total_duration_ns": int(payload.get("total_duration", 0) or 0),
            "llm_load_duration_ns": int(payload.get("load_duration", 0) or 0),
            "llm_prompt_eval_duration_ns": int(payload.get("prompt_eval_duration", 0) or 0),
            "llm_eval_duration_ns": int(payload.get("eval_duration", 0) or 0),
        }

        return self._parse(raw_text, w, h, confidence_threshold, llm_meta)

    # ── Private ───────────────────────────────────────────────────────────────

    def _call_ollama(self, image_b64: str, prompt: str) -> dict[str, Any]:
        payload = {
            "model":  self.model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "seed": 42},
        }
        resp = requests.post(self.api_url, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def _parse(
        self,
        text: str,
        img_w: int,
        img_h: int,
        threshold: float,
        meta: dict[str, Any],
    ) -> DetectionResult:
        text = re.sub(r"```json\s*|```", "", text).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group())
                except json.JSONDecodeError:
                    return DetectionResult(
                        found=False, box=None, confidence=0.0,
                        reason=f"LLM unparseable response: {text[:80]}", source="llm", meta=meta,
                    )
            else:
                return DetectionResult(
                    found=False, box=None, confidence=0.0,
                    reason=f"LLM unparseable response: {text[:80]}", source="llm", meta=meta,
                )

        found      = bool(data.get("found", False))
        confidence = float(data.get("confidence", 0.0))
        reason     = str(data.get("reason", ""))
        box_data   = data.get("box")

        if not found or box_data is None:
            return DetectionResult(
                found=False, box=None, confidence=confidence,
                reason=f"[LLM] {reason}", source="llm", meta=meta,
            )

        if confidence < threshold:
            return DetectionResult(
                found=False, box=None, confidence=confidence,
                reason=f"[LLM] conf {confidence:.3f} below threshold — {reason}",
                source="llm", meta=meta,
            )

        try:
            x1 = max(0,     int(box_data["x1_pct"] / 100 * img_w))
            y1 = max(0,     int(box_data["y1_pct"] / 100 * img_h))
            x2 = min(img_w, int(box_data["x2_pct"] / 100 * img_w))
            y2 = min(img_h, int(box_data["y2_pct"] / 100 * img_h))
        except (KeyError, TypeError, ValueError) as exc:
            return DetectionResult(
                found=False, box=None, confidence=confidence,
                reason=f"[LLM] bad box coords: {exc}", source="llm", meta=meta,
            )

        if x2 <= x1 or y2 <= y1:
            return DetectionResult(
                found=False, box=None, confidence=confidence,
                reason=f"[LLM] degenerate box ({x1},{y1},{x2},{y2})", source="llm", meta=meta,
            )

        return DetectionResult(
            found=True, box=(x1, y1, x2, y2), confidence=confidence,
            reason=f"[LLM] {reason}", source="llm",
            meta=meta,
        )

    def _check_ollama(self, base_url: str) -> bool:
        try:
            resp = requests.get(f"{base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            available  = [m["name"] for m in resp.json().get("models", [])]
            model_base = self.model.split(":")[0]
            pulled     = any(m.split(":")[0] == model_base for m in available)
            if not pulled:
                print(f"\n[WARNING] Ollama model '{self.model}' not found.")
                print(f"  Available : {available or 'none'}")
                print(f"  Run       : ollama pull {self.model}\n")
                return False
            else:
                print(f"[LLM ] Ollama model '{self.model}' is available.")
                return True
        except requests.exceptions.ConnectionError:
            print("\n[WARNING] Cannot reach Ollama. Run: ollama serve")
            return False
        except requests.exceptions.RequestException as exc:
            print(f"\n[WARNING] Ollama check failed: {exc}")
            return False


# ── Helpers ────────────────────────────────────────────────────────────────────

def _encode_image(pil_image: Image.Image) -> str:
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
