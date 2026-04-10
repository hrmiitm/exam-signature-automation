"""
Batch image processor for signature detection and cropping.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from detector import SignatureDetector
from detector_dino import DetectionResult

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class Processor:
    def __init__(
        self,
        output_dir: Path,
        confidence_threshold: float = 0.35,
        crop_padding: int = 15,
        model: str = "moondream:latest",
        ollama_url: str = "http://localhost:11434",
        device: str | None = None,
        batch_size: int = 8,
        io_workers: int = 8,
        llm_workers: int = 2,
        auto_batch_adjust: bool = False,
        max_image_side: int = 2200,
    ):
        self.output_dir = Path(output_dir)
        self.signatures_dir = self.output_dir / "signatures"
        self.flagged_dir = self.output_dir / "flagged"
        self.logs_dir = self.output_dir / "logs"

        self.confidence_threshold = confidence_threshold
        self.crop_padding = max(0, int(crop_padding))
        self.model = model
        self.ollama_url = ollama_url
        self.device = device
        self.requested_batch_size = max(1, int(batch_size))
        self.auto_batch_adjust = bool(auto_batch_adjust)
        self.max_image_side = max(0, int(max_image_side))
        cpu_count = os.cpu_count() or 4
        requested_io = max(1, int(io_workers or min(16, cpu_count)))
        requested_llm = max(1, int(llm_workers))
        self.requested_io_workers = requested_io
        self.requested_llm_workers = requested_llm
        self.io_workers = min(requested_io, max(2, cpu_count * 2))
        self.llm_workers = min(requested_llm, max(1, cpu_count))
        self.batch_size = self._safe_initial_batch_size(self.requested_batch_size)

        self._prepare_dirs()
        self.logger = self._build_logger()

        self.detector = SignatureDetector(
            confidence_threshold=confidence_threshold,
            device=device,
            model=model,
            ollama_url=ollama_url,
            llm_workers=self.llm_workers,
        )

        # If DINO falls back from requested device (e.g., CUDA OOM -> CPU),
        # recompute safe batch size for the actual device.
        effective_device = self.detector.dino.device
        if (self.device or "auto") != effective_device:
            self.logger.warning(
                "Detector device fallback: requested=%s effective=%s",
                self.device or "auto",
                effective_device,
            )
            self.device = effective_device
            self.batch_size = self._safe_initial_batch_size(self.requested_batch_size)

    def run(self, input_dir: Path) -> None:
        started_at = datetime.now(timezone.utc)
        run_start = time.perf_counter()

        image_paths = self._collect_images(input_dir)
        total_images = len(image_paths)

        if total_images == 0:
            self.logger.warning("No image files found in input directory: %s", input_dir)
            print("[WARN] No images found. Nothing to process.")
            return

        self.logger.info(
            "Starting processing: images=%d batch_size=%d requested_batch=%d io_workers=%d llm_workers=%d auto_batch_adjust=%s max_image_side=%s",
            total_images,
            self.batch_size,
            self.requested_batch_size,
            self.io_workers,
            self.llm_workers,
            self.auto_batch_adjust,
            self.max_image_side if self.max_image_side > 0 else "disabled",
        )

        loading_elapsed = 0.0
        adjusted_batch_events = 0
        resized_images = 0

        summary_rows: list[dict[str, object]] = []
        per_image_records: list[dict[str, object]] = []
        flagged_names: list[str] = []

        detected_count = 0
        flagged_count = 0
        error_count = 0
        source_counts = {"dino": 0, "llm": 0, "heuristic": 0, "unknown": 0, "error": 0}

        detect_started = time.perf_counter()
        current_batch_size = self.batch_size
        idx = 0
        progress = tqdm(total=total_images, desc="Detecting", unit="img")

        while idx < total_images:
            batch_paths = image_paths[idx: idx + current_batch_size]
            batch_load_started = time.perf_counter()
            batch_records = self._load_images_parallel(batch_paths, input_dir)
            loading_elapsed += time.perf_counter() - batch_load_started
            resized_images += sum(1 for r in batch_records if bool(r.get("resized", False)))
            valid_records = [r for r in batch_records if r["image"] is not None]

            for record in batch_records:
                if record["image"] is not None:
                    continue

                rel_name = str(record["rel_name"])
                error_count += 1
                flagged_count += 1
                source_counts["error"] += 1
                flagged_names.append(rel_name)
                self._save_flagged(Path(record["path"]), rel_name)

                row = {
                    "file_name": rel_name,
                    "status": "error",
                    "source": "error",
                    "confidence": 0.0,
                    "x1": "",
                    "y1": "",
                    "x2": "",
                    "y2": "",
                    "reason": f"load error: {record['load_error']}",
                    "crop_path": "",
                    "load_seconds": round(float(record["load_seconds"]), 4),
                    "detect_seconds_est": 0.0,
                    "save_seconds": 0.0,
                    "processing_seconds": round(float(record["load_seconds"]), 4),
                    "resized": bool(record.get("resized", False)),
                    "original_width": record.get("original_width", ""),
                    "original_height": record.get("original_height", ""),
                    "final_width": record.get("final_width", ""),
                    "final_height": record.get("final_height", ""),
                    "llm_prompt_tokens": 0,
                    "llm_completion_tokens": 0,
                    "llm_total_tokens": 0,
                    "llm_total_duration_ms": 0.0,
                    "llm_eval_duration_ms": 0.0,
                }
                summary_rows.append(row)
                json_record = dict(row)
                json_record["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
                per_image_records.append(json_record)
                self.logger.error("file=%s status=error reason=%s", rel_name, row["reason"])

            if not valid_records:
                idx += len(batch_paths)
                progress.update(len(batch_paths))
                continue

            batch_images = [record["image"] for record in valid_records]
            batch_detect_started = time.perf_counter()

            try:
                batch_results = self.detector.detect_batch(batch_images)
            except Exception as exc:
                if self.auto_batch_adjust and current_batch_size > 1:
                    old_size = current_batch_size
                    current_batch_size = max(1, current_batch_size // 2)
                    adjusted_batch_events += 1
                    for record in valid_records:
                        record["image"] = None
                    self.logger.warning(
                        "Batch failure at idx=%d with size=%d (%s). Retrying with size=%d",
                        idx,
                        old_size,
                        type(exc).__name__,
                        current_batch_size,
                    )
                    continue

                self.logger.exception("Batch detection failed, falling back to single-image detection")
                batch_results = []
                for image in batch_images:
                    try:
                        batch_results.append(self.detector.detect(image))
                    except Exception as single_exc:
                        batch_results.append(
                            DetectionResult(
                                found=False,
                                box=None,
                                confidence=0.0,
                                reason=f"detection error: {single_exc}",
                                source="error",
                                meta={},
                            )
                        )
                self.logger.error("Batch exception: %s", exc)

            batch_detect_elapsed = time.perf_counter() - batch_detect_started
            detect_per_image_est = batch_detect_elapsed / max(1, len(valid_records))

            for record, result in zip(valid_records, batch_results):
                rel_name = str(record["rel_name"])
                image_path = Path(record["path"])
                image = record["image"]
                save_started = time.perf_counter()

                source = result.source if result.source in source_counts else "unknown"
                source_counts[source] += 1

                crop_path = ""
                status = "flagged"
                box = result.box

                if result.found and box is not None:
                    cropped = self._crop_with_padding(image, box)
                    crop_path = str(self._save_signature(cropped, rel_name))
                    status = "detected"
                    detected_count += 1
                else:
                    self._save_flagged(image_path, rel_name)
                    flagged_names.append(rel_name)
                    flagged_count += 1

                save_elapsed = time.perf_counter() - save_started
                load_elapsed = float(record["load_seconds"])
                total_elapsed = round(load_elapsed + detect_per_image_est + save_elapsed, 4)
                meta = result.meta or {}

                row = {
                    "file_name": rel_name,
                    "status": status,
                    "source": result.source,
                    "confidence": round(float(result.confidence), 6),
                    "x1": box[0] if box else "",
                    "y1": box[1] if box else "",
                    "x2": box[2] if box else "",
                    "y2": box[3] if box else "",
                    "reason": result.reason,
                    "crop_path": crop_path,
                    "load_seconds": round(load_elapsed, 4),
                    "detect_seconds_est": round(detect_per_image_est, 4),
                    "save_seconds": round(save_elapsed, 4),
                    "processing_seconds": total_elapsed,
                    "resized": bool(record.get("resized", False)),
                    "original_width": record.get("original_width", ""),
                    "original_height": record.get("original_height", ""),
                    "final_width": record.get("final_width", ""),
                    "final_height": record.get("final_height", ""),
                    "llm_prompt_tokens": int(meta.get("llm_prompt_tokens", 0) or 0),
                    "llm_completion_tokens": int(meta.get("llm_completion_tokens", 0) or 0),
                    "llm_total_tokens": int(meta.get("llm_prompt_tokens", 0) or 0)
                    + int(meta.get("llm_completion_tokens", 0) or 0),
                    "llm_total_duration_ms": self._ns_to_ms(meta.get("llm_total_duration_ns", 0)),
                    "llm_eval_duration_ms": self._ns_to_ms(meta.get("llm_eval_duration_ns", 0)),
                }
                summary_rows.append(row)

                json_record = dict(row)
                json_record["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
                per_image_records.append(json_record)

                self.logger.info(
                    "file=%s status=%s source=%s conf=%.3f load=%.3fs detect=%.3fs save=%.3fs reason=%s",
                    rel_name,
                    status,
                    result.source,
                    result.confidence,
                    load_elapsed,
                    detect_per_image_est,
                    save_elapsed,
                    result.reason,
                )

                record["image"] = None

            idx += len(batch_paths)
            progress.update(len(batch_paths))

        progress.close()

        detection_elapsed = time.perf_counter() - detect_started

        write_started = time.perf_counter()
        self._write_summary_csv(summary_rows)
        self._write_flagged_file(flagged_names)
        self._write_jsonl(per_image_records)
        write_elapsed = time.perf_counter() - write_started

        finished_at = datetime.now(timezone.utc)
        run_elapsed_s = round(time.perf_counter() - run_start, 4)

        token_prompt_sum = sum(int(r["llm_prompt_tokens"]) for r in summary_rows)
        token_completion_sum = sum(int(r["llm_completion_tokens"]) for r in summary_rows)
        llm_total_ms = round(sum(float(r["llm_total_duration_ms"]) for r in summary_rows), 3)
        llm_eval_ms = round(sum(float(r["llm_eval_duration_ms"]) for r in summary_rows), 3)

        stats = {
            "run": {
                "started_at_utc": started_at.isoformat(),
                "finished_at_utc": finished_at.isoformat(),
                "elapsed_seconds": run_elapsed_s,
                "images_per_second": round(total_images / run_elapsed_s, 4) if run_elapsed_s > 0 else 0.0,
            },
            "performance": {
                "loading_seconds": round(loading_elapsed, 4),
                "detection_seconds": round(detection_elapsed, 4),
                "write_seconds": round(write_elapsed, 4),
            },
            "parallelism": {
                "requested_batch_size": self.requested_batch_size,
                "batch_size": self.batch_size,
                "effective_batch_size_last": current_batch_size,
                "auto_batch_adjust": self.auto_batch_adjust,
                "auto_batch_adjust_events": adjusted_batch_events,
                "requested_io_workers": self.requested_io_workers,
                "io_workers": self.io_workers,
                "requested_llm_workers": self.requested_llm_workers,
                "llm_workers": self.llm_workers,
            },
            "config": {
                "input_dir": str(input_dir),
                "output_dir": str(self.output_dir),
                "model": self.model,
                "ollama_url": self.ollama_url,
                "device": self.device or "auto",
                "confidence_threshold": self.confidence_threshold,
                "crop_padding": self.crop_padding,
                "max_image_side": self.max_image_side,
            },
            "counts": {
                "total_images": total_images,
                "detected": detected_count,
                "flagged": flagged_count,
                "errors": error_count,
                "resized_images": resized_images,
            },
            "sources": source_counts,
            "llm_usage": {
                "prompt_tokens": token_prompt_sum,
                "completion_tokens": token_completion_sum,
                "total_tokens": token_prompt_sum + token_completion_sum,
                "total_duration_ms": llm_total_ms,
                "eval_duration_ms": llm_eval_ms,
            },
            "artifacts": {
                "summary_csv": str(self.output_dir / "summary.csv"),
                "flagged_txt": str(self.output_dir / "flagged.txt"),
                "stats_json": str(self.output_dir / "stats.json"),
                "processing_log": str(self.logs_dir / "processing.log"),
                "per_image_jsonl": str(self.logs_dir / "per_image.jsonl"),
            },
        }

        stats_path = self.output_dir / "stats.json"
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

        print("\n" + "=" * 70)
        print("Run Summary")
        print("=" * 70)
        print(f"Total images       : {total_images}")
        print(f"Detected signatures: {detected_count}")
        print(f"Flagged images     : {flagged_count}")
        print(f"Errors             : {error_count}")
        print(f"Resized images     : {resized_images}")
        print(f"Elapsed (s)        : {run_elapsed_s}")
        print(f"Loading (s)        : {loading_elapsed:.4f}")
        print(f"Detection (s)      : {detection_elapsed:.4f}")
        print(f"Write (s)          : {write_elapsed:.4f}")
        print(f"LLM total tokens   : {token_prompt_sum + token_completion_sum}")
        print(f"LLM total time (ms): {llm_total_ms}")
        print(f"Batch size used    : {self.batch_size} (requested {self.requested_batch_size})")
        print(f"Batch adjustments  : {adjusted_batch_events}")
        print(f"Throughput (img/s) : {stats['run']['images_per_second']}")
        print(f"Output directory   : {self.output_dir}")
        print("=" * 70)

        self.logger.info("Completed run in %.3fs", run_elapsed_s)

    def _prepare_dirs(self) -> None:
        self.signatures_dir.mkdir(parents=True, exist_ok=True)
        self.flagged_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger("signature_detector")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        log_path = self.logs_dir / "processing.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        stream_handler = logging.StreamHandler()

        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler.setFormatter(fmt)
        stream_handler.setFormatter(fmt)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        return logger

    def _collect_images(self, input_dir: Path) -> list[Path]:
        files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        return sorted(files)

    def _load_images_parallel(self, image_paths: list[Path], input_dir: Path) -> list[dict[str, object]]:
        with ThreadPoolExecutor(max_workers=self.io_workers) as pool:
            records = list(pool.map(lambda p: self._load_single_image(p, input_dir), image_paths))
        return records

    def _load_single_image(self, image_path: Path, input_dir: Path) -> dict[str, object]:
        start = time.perf_counter()
        rel_name = str(image_path.relative_to(input_dir))
        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
            orig_w, orig_h = image.size
            image, resized = self._resize_if_needed(image)
            final_w, final_h = image.size
            return {
                "path": image_path,
                "rel_name": rel_name,
                "image": image,
                "load_error": "",
                "resized": resized,
                "original_width": orig_w,
                "original_height": orig_h,
                "final_width": final_w,
                "final_height": final_h,
                "load_seconds": round(time.perf_counter() - start, 6),
            }
        except Exception as exc:
            return {
                "path": image_path,
                "rel_name": rel_name,
                "image": None,
                "load_error": str(exc),
                "resized": False,
                "original_width": "",
                "original_height": "",
                "final_width": "",
                "final_height": "",
                "load_seconds": round(time.perf_counter() - start, 6),
            }

    def _resize_if_needed(self, image: Image.Image) -> tuple[Image.Image, bool]:
        if self.max_image_side <= 0:
            return image, False

        width, height = image.size
        longest = max(width, height)
        if longest <= self.max_image_side:
            return image, False

        scale = self.max_image_side / float(longest)
        new_w = max(1, int(width * scale))
        new_h = max(1, int(height * scale))
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS), True

    def _safe_initial_batch_size(self, requested: int) -> int:
        """Apply conservative caps to avoid OS-level OOM kills before retries can happen."""
        device = (self.device or "auto").lower()

        # CPU inference memory rises quickly with image resolution; keep safer default cap.
        if device == "cpu":
            if self.max_image_side == 0 or self.max_image_side > 1800:
                return min(requested, 1)
            if self.max_image_side > 1400:
                return min(requested, 2)
            return min(requested, 4)

        # CUDA can usually handle larger batches, but cap to avoid large spikes.
        if device == "cuda":
            if self.max_image_side == 0 or self.max_image_side > 2200:
                return min(requested, 4)
            if self.max_image_side > 1600:
                return min(requested, 8)
            return min(requested, 16)

        # Auto/MPS/default: conservative middle ground.
        return min(requested, 4)

    def _crop_with_padding(self, image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
        x1, y1, x2, y2 = box
        width, height = image.size

        x1 = max(0, x1 - self.crop_padding)
        y1 = max(0, y1 - self.crop_padding)
        x2 = min(width, x2 + self.crop_padding)
        y2 = min(height, y2 + self.crop_padding)

        return image.crop((x1, y1, x2, y2))

    def _save_signature(self, cropped: Image.Image, rel_name: str) -> Path:
        rel_path = Path(rel_name)
        stem = rel_path.stem
        if stem.lower().endswith("_signature"):
            out_name = f"{stem}{rel_path.suffix}"
        else:
            out_name = f"{stem}_signature{rel_path.suffix}"
        out_path = self.signatures_dir / out_name
        cropped.save(out_path)
        return out_path

    def _save_flagged(self, image_path: Path, rel_name: str) -> Path:
        out_path = self.flagged_dir / rel_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, out_path)
        return out_path

    def _write_summary_csv(self, rows: list[dict[str, object]]) -> None:
        csv_path = self.output_dir / "summary.csv"
        fields = [
            "file_name",
            "status",
            "source",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "reason",
            "crop_path",
            "load_seconds",
            "detect_seconds_est",
            "save_seconds",
            "processing_seconds",
            "resized",
            "original_width",
            "original_height",
            "final_width",
            "final_height",
            "llm_prompt_tokens",
            "llm_completion_tokens",
            "llm_total_tokens",
            "llm_total_duration_ms",
            "llm_eval_duration_ms",
        ]

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def _write_flagged_file(self, names: list[str]) -> None:
        flagged_txt = self.output_dir / "flagged.txt"
        names_sorted = sorted(set(names))
        flagged_txt.write_text("\n".join(names_sorted) + ("\n" if names_sorted else ""), encoding="utf-8")

    def _write_jsonl(self, records: list[dict[str, object]]) -> None:
        out_path = self.logs_dir / "per_image.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=True) + "\n")

    @staticmethod
    def _ns_to_ms(value_ns: object) -> float:
        try:
            return round(int(value_ns) / 1_000_000, 3)
        except (TypeError, ValueError):
            return 0.0
