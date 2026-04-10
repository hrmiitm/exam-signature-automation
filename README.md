
# RUN

```bash
python3 main.py --input original_signatures/ --device cpu --batch-size 4 --io-workers 4 --llm-workers 2 --auto-batch-adjust --max-image-side 1600;
```

```bash
python3 main.py --input test --output test_output --device cpu --batch-size 4 --io-workers 4 --llm-workers 2 --auto-batch-adjust --max-image-side 1600 --threshold 0.45
```


## Command Parameters Explained

**Parameter Explanations and Performance Tuning:**

- `--input <folder>`
	- **What it does:** Path to the folder containing images to process.
	- **How to adjust:** Set to your image directory. More images = longer processing.

- `--device <cpu|cuda|mps>`
	- **What it does:** Sets the inference device. Options: `cpu`, `cuda` (NVIDIA GPU), `mps` (Apple Silicon).
	- **How to adjust:** Use `cuda` for fastest performance if you have a compatible GPU. Use `cpu` if no GPU or for stability.

- `--batch-size <int>`
	- **What it does:** Number of images processed in parallel by the DINO model.
	- **How to adjust:** Increase for faster processing on powerful GPUs. Lower if you get out-of-memory errors or on CPU.

- `--io-workers <int>`
	- **What it does:** Number of parallel workers for loading images from disk.
	- **How to adjust:** Set to number of CPU cores for best disk throughput. Too high can cause instability.

- `--llm-workers <int>`
	- **What it does:** Number of parallel workers for LLM verification (for uncertain images).
	- **How to adjust:** Increase for faster LLM processing if you have enough CPU/RAM. Too high can overload system or LLM server.

- `--auto-batch-adjust`
	- **What it does:** Automatically reduces batch size if memory/runtime errors occur.
	- **How to adjust:** Keep enabled for stability, especially on limited hardware.

- `--max-image-side <int>`
	- **What it does:** Downscales images so the largest side is at most this many pixels.
	- **How to adjust:** Lower for less memory use and faster processing, but may reduce detection accuracy. Raise for higher quality if you have enough memory.

**Performance Tips:**
- For best speed, use `--device cuda` and increase `--batch-size` as high as your GPU allows without crashing.
- For stability, keep `--auto-batch-adjust` enabled and set `--max-image-side` to 1400–2200.
- Tune `--io-workers` and `--llm-workers` based on your CPU core count and system RAM.
- If you get out-of-memory errors, lower `--batch-size` or `--max-image-side`.

