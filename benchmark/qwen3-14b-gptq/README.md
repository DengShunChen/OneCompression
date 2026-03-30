# Qwen3-14B GPTQ Benchmark

GPTQ benchmark for [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) using OneComp v0.3.7+jointq.

All combinations of `bits × group_size` are run in a single pass, sharing calibration data accumulation across quantizers for efficiency.

Two configurations are benchmarked:

1. **GPTQ (default)** — `actorder=false`, `mse=false`
2. **GPTQ (mse+actorder)** — `actorder=true`, `mse=true` (strongest GPTQ setting)

## Benchmark Configuration

### Common Parameters

| Parameter | Values |
|---|---|
| bits | 4, 3 |
| group_size | 128, per-channel |
| symmetric | true |
| num_calibration_samples | 1024 |
| calibration_strategy | drop_rand |
| max_length | 2048 |

This produces **4 quantizers** (2 bits × 2 group sizes) per configuration.

### Configuration-Specific Parameters

| Parameter | default | mse+actorder |
|---|---|---|
| actorder | false | true |
| mse | false | true |

### Evaluation

- Perplexity (WikiText-2)
- Accuracy (lm-eval-harness)

Both are computed for the original (unquantized) model and all quantized models.

## Usage

Requires [Hydra](https://hydra.cc/) (see [benchmark/README.md](../README.md) for installation).

```bash
# default
python quant_benchmark.py model_path=/path/to/Qwen3-14B

# mse+actorder
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    gptq.actorder=true gptq.mse=true output_dir=qwen3-14b-mse-actorder
```

### Hydra Overrides

You can override any parameter from the command line:

```bash
# Run only 4-bit
python quant_benchmark.py model_path=/path/to/model 'gptq.bits=[4]'

# Change calibration samples
python quant_benchmark.py model_path=/path/to/model num_calibration_samples=512
```

## Results

### GPTQ (default)

PPL = perplexity on WikiText-2 (↓ lower is better). Accuracy = 0-shot `acc_norm` where available, `acc` otherwise (winogrande) (↑ higher is better).

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| — (Original) | — | 8.64 | 0.6032 | 0.8283 | 0.7971 | 0.7293 | — |
| 4 | 128 | 8.85 | 0.5947 | 0.8182 | 0.7982 | 0.7316 | 481.9 |
| 4 | per-channel | 9.15 | 0.5802 | 0.8056 | 0.7873 | 0.7056 | 470.1 |
| 3 | 128 | 10.10 | 0.5307 | 0.7727 | 0.7873 | 0.7001 | 480.1 |
| 3 | per-channel | 13.72 | 0.3976 | 0.5745 | 0.7356 | 0.6172 | 468.4 |

Total elapsed time (including calibration data preparation): 7414.3 s (~124 min).

### GPTQ (mse+actorder)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| — (Original) | — | 8.64 | 0.6032 | 0.8283 | 0.7971 | 0.7293 | — |
| 4 | 128 | 8.89 | 0.6101 | 0.8300 | 0.7911 | 0.7277 | 2288.2 |
| 4 | per-channel | 9.26 | 0.5922 | 0.8249 | 0.8025 | 0.7261 | 611.3 |
| 3 | 128 | 9.44 | 0.5811 | 0.8047 | 0.7933 | 0.7056 | 2694.7 |
| 3 | per-channel | 15.52 | 0.4855 | 0.7311 | 0.7677 | 0.6543 | 616.8 |

Total elapsed time (including calibration data preparation): 11776.3 s (~196 min).

## Environment

- GPU: NVIDIA B200 × 2

## Notes

This benchmark internally uses `Runner.quantize_with_calibration_chunked`, which can run multiple quantizers simultaneously without QEP. However, it requires the entire model to fit on the GPU and involves redundant forward passes. Addressing these limitations is future work.
