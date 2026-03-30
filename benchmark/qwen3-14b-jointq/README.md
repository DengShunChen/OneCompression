# Qwen3-14B JointQ Benchmark

JointQ benchmark for [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) using OneComp v0.3.7+jointq.

All combinations of `bits × group_size` are run in a single pass, sharing calibration data accumulation across quantizers for efficiency.

Four configurations are benchmarked:

1. **JointQ (default)** — `λ=0.2`, `actorder=false`
2. **JointQ (actorder)** — `λ=0.2`, `actorder=true`
3. **JointQ (noreg)** — `λ=0.0`, `actorder=false`
4. **JointQ (noreg+actorder)** — `λ=0.0`, `actorder=true`

## Benchmark Configuration

### Common Parameters

| Parameter | Values |
|---|---|
| bits | 4, 3 |
| group_size | 128, null (per-channel) |
| symmetric | true |
| num_calibration_samples | 1024 |
| calibration_strategy | drop_rand |
| max_length | 2048 |

This produces **4 quantizers** (2 bits × 2 group sizes) per configuration.

### Configuration-Specific Parameters

| Parameter | default | actorder | noreg | noreg+actorder |
|---|---|---|---|---|
| regularization_lambda | 0.2 | 0.2 | 0.0 | 0.0 |
| actorder | false | true | false | true |

### Evaluation

- Perplexity (WikiText-2)
- Accuracy (lm-eval-harness)

Both are computed for the original (unquantized) model and all quantized models.

## Usage

Requires [Hydra](https://hydra.cc/) (see [benchmark/README.md](../README.md) for installation).

```bash
# default (λ=0.2)
python quant_benchmark.py model_path=/path/to/Qwen3-14B

# actorder (λ=0.2)
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    jointq.actorder=true output_dir=qwen3-14b-actorder

# noreg (λ=0.0)
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    jointq.regularization_lambda=0.0 output_dir=qwen3-14b-noreg

# noreg+actorder (λ=0.0)
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    jointq.regularization_lambda=0.0 jointq.actorder=true output_dir=qwen3-14b-noreg-actorder
```

### Hydra Overrides

You can override any parameter from the command line:

```bash
# Run only 4-bit
python quant_benchmark.py model_path=/path/to/model 'jointq.bits=[4]'

# Change calibration samples
python quant_benchmark.py model_path=/path/to/model num_calibration_samples=512
```

## Results

PPL = perplexity on WikiText-2 (↓ lower is better). Accuracy = 0-shot `acc_norm` where available, `acc` otherwise (winogrande) (↑ higher is better).

### JointQ (λ=0.2, actorder=false)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| — (Original) | — | 8.64 | 0.6032 | 0.8283 | 0.7971 | 0.7293 | — |
| 4 | 128 | 8.88 | 0.6007 | 0.8215 | 0.7992 | 0.7332 | 3024.7 |
| 4 | per-channel | 9.96 | 0.5845 | 0.8110 | 0.7873 | 0.7080 | 5545.2 |
| 3 | 128 | 9.99 | 0.5401 | 0.7761 | 0.7791 | 0.7088 | 4936.1 |
| 3 | per-channel | 23.55 | 0.4727 | 0.7247 | 0.7546 | 0.6504 | 5632.7 |

Total elapsed time (including calibration data preparation): 24733.0 s (~412 min).

### JointQ (λ=0.2, actorder=true)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| — (Original) | — | 8.64 | 0.6032 | 0.8283 | 0.7971 | 0.7293 | — |
| 4 | 128 | 8.90 | 0.5939 | 0.8136 | 0.7878 | 0.7159 | 2731.9 |
| 4 | per-channel | 9.74 | 0.5700 | 0.8114 | 0.7873 | 0.7135 | 5594.3 |
| 3 | 128 | 10.18 | 0.5998 | 0.8182 | 0.7884 | 0.7238 | 4720.4 |
| 3 | per-channel | 23.12 | 0.4718 | 0.7100 | 0.7524 | 0.6582 | 5625.0 |

Total elapsed time (including calibration data preparation): 24286.7 s (~405 min).

### JointQ (λ=0.0, actorder=false)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| — (Original) | — | 8.64 | 0.6032 | 0.8283 | 0.7971 | 0.7293 | — |
| 4 | 128 | 8.87 | 0.6118 | 0.8253 | 0.8003 | 0.7277 | 7798.3 |
| 4 | per-channel | 9.47 | 0.5913 | 0.8178 | 0.7922 | 0.7174 | 3246.4 |
| 3 | 128 | 9.85 | 0.5725 | 0.7980 | 0.7867 | 0.7143 | 7025.6 |
| 3 | per-channel | 20.48 | 0.5290 | 0.7774 | 0.7688 | 0.7198 | 4323.2 |

Total elapsed time (including calibration data preparation): 27944.6 s (~466 min).

### JointQ (λ=0.0, actorder=true)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| — (Original) | — | 8.64 | 0.6032 | 0.8283 | 0.7971 | 0.7293 | — |
| 4 | 128 | 8.86 | 0.5973 | 0.8321 | 0.8041 | 0.7245 | 2599.7 |
| 4 | per-channel | 9.39 | 0.6058 | 0.8173 | 0.7976 | 0.7127 | 3682.9 |
| 3 | 128 | 10.59 | 0.5819 | 0.7934 | 0.7911 | 0.7293 | 3156.6 |
| 3 | per-channel | 18.43 | 0.5350 | 0.7858 | 0.7699 | 0.7245 | 4549.0 |

Total elapsed time (including calibration data preparation): 19601.6 s (~327 min).

## Environment

- GPU: NVIDIA B200 × 2

## Output

Results are saved under the `output_dir` directory (default: `qwen3-14b/`):

- `quantization_statistics_<quantizer_name>.json` — per-layer quantization error statistics
- Perplexity and accuracy results are printed to stdout

## Notes

This benchmark internally uses `Runner.quantize_with_calibration_chunked`, which can run multiple quantizers simultaneously without QEP. However, it requires the entire model to fit on the GPU and involves redundant forward passes. Addressing these limitations is future work.
