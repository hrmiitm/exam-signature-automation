"""
Signature Detector — main entry point.

Usage:
    python main.py --input /path/to/images
    python main.py --input /path/to/images --model llava:13b
    python main.py --input /path/to/images --model moondream:latest --device cuda
    python main.py --input /path/to/images --output /path/to/output --threshold 0.4 --padding 20
    python main.py --input /path/to/images --device cuda --batch-size 16 --io-workers 16 --llm-workers 4
    python main.py --input /path/to/images --device cuda --batch-size 16 --auto-batch-adjust --max-image-side 2200
"""

import argparse
import sys
from pathlib import Path

# Ensure the directory containing main.py is on the path,
# regardless of where the script is invoked from.
sys.path.insert(0, str(Path(__file__).parent))

from processor import Processor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect and crop signatures from document images using a local Ollama vision model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models (pull before use):
    ollama pull moondream      <- default: smallest and fastest (1.6 GB)
    ollama pull minicpm-v      <- fast, great at documents (5 GB)
  ollama pull llava:13b      <- more accurate, slower (8 GB)
  ollama pull llava:34b      <- best accuracy, needs 20 GB RAM

Output structure:
  output/
  ├── signatures/   <- cropped signature images
  ├── flagged/      <- originals with no or unclear signature
    ├── logs/
    │   ├── processing.log  <- detailed runtime logs
    │   └── per_image.jsonl <- one JSON record per image
  ├── flagged.txt   <- plain list of flagged filenames
  └── summary.csv   <- full per-image report with confidence scores
    └── stats.json    <- aggregate stats, token counts and timings
        """,
    )
    parser.add_argument("--input",  "-i", required=True, type=str,
        help="Path to folder containing images to process")
    parser.add_argument("--output", "-o", type=str, default="output",
        help="Path to output folder (default: ./output)")
    parser.add_argument("--model",  "-m", type=str, default="moondream:latest",
        help="Ollama vision model name (default: moondream:latest)")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)")
    parser.add_argument("--device", type=str, default=None,
        help="Inference device: cpu, cuda, mps (default: auto)")
    parser.add_argument("--threshold", "-t", type=float, default=0.35,
        help="Min confidence to accept a detection (default: 0.35)")
    parser.add_argument("--padding", "-p", type=int, default=15,
        help="Pixels of padding around the cropped signature (default: 15)")
    parser.add_argument("--batch-size", type=int, default=8,
        help="Batch size for DINO inference (higher usually faster on GPU, default: 8)")
    parser.add_argument("--io-workers", type=int, default=8,
        help="Parallel workers for CPU image loading (default: 8)")
    parser.add_argument("--llm-workers", type=int, default=2,
        help="Parallel workers for uncertain-image LLM verification (default: 2)")
    parser.add_argument("--auto-batch-adjust", action="store_true", default=True,
        help="Automatically reduce batch size on memory/runtime failures (default: enabled)")
    parser.add_argument("--no-auto-batch-adjust", action="store_false", dest="auto_batch_adjust",
        help="Disable automatic batch-size reduction")
    parser.add_argument("--max-image-side", type=int, default=1600,
        help="Auto-downscale images larger than this side length (0 disables, default: 1600)")
    return parser.parse_args()


def main():
    args = parse_args()

    input_path  = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        print(f"[ERROR] Input path does not exist: {input_path}")
        sys.exit(1)
    if not input_path.is_dir():
        print(f"[ERROR] Input path is not a directory: {input_path}")
        sys.exit(1)

    print("=" * 60)
    print("  Signature Detector  (Ollama backend)")
    print("=" * 60)
    print(f"  Input      : {input_path}")
    print(f"  Output     : {output_path}")
    print(f"  Model      : {args.model}")
    print(f"  Ollama URL : {args.ollama_url}")
    print(f"  Device     : {args.device or 'auto'}")
    print(f"  Threshold  : {args.threshold}")
    print(f"  Padding    : {args.padding}px")
    print(f"  Batch size : {args.batch_size}")
    print(f"  IO workers : {args.io_workers}")
    print(f"  LLM workers: {args.llm_workers}")
    print(f"  Auto batch : {args.auto_batch_adjust}")
    print(f"  Max side   : {args.max_image_side if args.max_image_side > 0 else 'disabled'}")
    print("  Note       : worker counts may be auto-capped for stability")
    print("=" * 60)

    processor = Processor(
        output_dir=output_path,
        confidence_threshold=args.threshold,
        crop_padding=args.padding,
        model=args.model,
        ollama_url=args.ollama_url,
        device=args.device,
        batch_size=args.batch_size,
        io_workers=args.io_workers,
        llm_workers=args.llm_workers,
        auto_batch_adjust=args.auto_batch_adjust,
        max_image_side=args.max_image_side,
    )
    processor.run(input_path)


if __name__ == "__main__":
    main()
