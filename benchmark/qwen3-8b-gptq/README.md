# Qwen3-8B GPTQ Benchmark

GPTQ benchmark for [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) using OneComp v0.3.7+jointq.

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
python quant_benchmark.py model_path=/path/to/Qwen3-8B

# mse+actorder
python quant_benchmark.py model_path=/path/to/Qwen3-8B \
    gptq.actorder=true gptq.mse=true output_dir=qwen3-8b-mse-actorder
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
| — (Original) | — | 9.72 | 0.5657 | 0.8093 | 0.7775 | 0.6756 | — |
| 4 | 128 | 10.29 | 0.5538 | 0.7946 | 0.7677 | 0.6709 | 275.8 |
| 4 | per-channel | 10.97 | 0.5085 | 0.7412 | 0.7688 | 0.6693 | 269.0 |
| 3 | 128 | 11.71 | 0.4966 | 0.7273 | 0.7486 | 0.6440 | 275.2 |
| 3 | per-channel | 20.21 | 0.3234 | 0.4293 | 0.6806 | 0.5501 | 269.1 |

Total elapsed time (including calibration data preparation): 3949.8 s (~66 min).

### GPTQ (mse+actorder)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| — (Original) | — | 9.72 | 0.5657 | 0.8093 | 0.7775 | 0.6756 | — |
| 4 | 128 | 9.81 | 0.5538 | 0.8035 | 0.7775 | 0.6859 | 1433.3 |
| 4 | per-channel | 11.18 | 0.5410 | 0.7803 | 0.7791 | 0.6693 | 371.1 |
| 3 | 128 | 11.29 | 0.5034 | 0.7609 | 0.7633 | 0.6969 | 1688.5 |
| 3 | per-channel | 42.71 | 0.3157 | 0.4407 | 0.6801 | 0.5604 | 376.2 |

Total elapsed time (including calibration data preparation): 6715.4 s (~112 min).

## Environment

- GPU: NVIDIA B200 × 1

## Notes

This benchmark internally uses `Runner.quantize_with_calibration_chunked`, which can run multiple quantizers simultaneously without QEP. However, it requires the entire model to fit on the GPU and involves redundant forward passes. Addressing these limitations is future work.
