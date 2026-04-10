
# RUN

```bash
python3 main.py --input original_signatures/ --device cpu --batch-size 4 --io-workers 4 --llm-workers 2 --auto-batch-adjust --max-image-side 1600;
```

```bash
python3 main.py --input test --output test_output --device cpu --batch-size 4 --io-workers 4 --llm-workers 2 --auto-batch-adjust --max-image-side 1600 --threshold 0.45
```

