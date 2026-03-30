# Qwen3-8B JointQ Benchmark

JointQ benchmark for [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) using OneComp v0.3.7+jointq.

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
python quant_benchmark.py model_path=/path/to/Qwen3-8B

# actorder (λ=0.2)
python quant_benchmark.py model_path=/path/to/Qwen3-8B \
    jointq.actorder=true output_dir=qwen3-8b-actorder

# noreg (λ=0.0)
python quant_benchmark.py model_path=/path/to/Qwen3-8B \
    jointq.regularization_lambda=0.0 output_dir=qwen3-8b-noreg

# noreg+actorder (λ=0.0)
python quant_benchmark.py model_path=/path/to/Qwen3-8B \
    jointq.regularization_lambda=0.0 jointq.actorder=true output_dir=qwen3-8b-noreg-actorder
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
| — (Original) | — | 9.72 | 0.5657 | 0.8093 | 0.7775 | 0.6756 | — |
| 4 | 128 | 10.29 | 0.5469 | 0.7849 | 0.7650 | 0.6819 | 1183.8 |
| 4 | per-channel | 11.33 | 0.4974 | 0.7483 | 0.7617 | 0.6330 | 2334.3 |
| 3 | 128 | 12.38 | 0.4966 | 0.7395 | 0.7568 | 0.6772 | 1895.9 |
| 3 | per-channel | 42.10 | 0.2986 | 0.4987 | 0.6779 | 0.5478 | 2443.9 |

Total elapsed time (including calibration data preparation): 10807.6 s (~180 min).

### JointQ (λ=0.2, actorder=true)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| — (Original) | — | 9.72 | 0.5657 | 0.8093 | 0.7775 | 0.6756 | — |
| 4 | 128 | 10.12 | 0.5546 | 0.8060 | 0.7666 | 0.6914 | 1109.4 |
| 4 | per-channel | 11.41 | 0.4906 | 0.7517 | 0.7579 | 0.6433 | 2368.4 |
| 3 | 128 | 13.09 | 0.4795 | 0.7130 | 0.7535 | 0.6622 | 1849.4 |
| 3 | per-channel | 41.73 | 0.2927 | 0.5025 | 0.6752 | 0.5572 | 2451.3 |

Total elapsed time (including calibration data preparation): 10760.7 s (~179 min).

### JointQ (λ=0.0, actorder=false)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| — (Original) | — | 9.72 | 0.5657 | 0.8093 | 0.7775 | 0.6756 | — |
| 4 | 128 | 10.13 | 0.5546 | 0.7992 | 0.7682 | 0.6843 | 3794.9 |
| 4 | per-channel | 10.67 | 0.5034 | 0.7290 | 0.7786 | 0.6922 | 1245.1 |
| 3 | 128 | 11.43 | 0.5307 | 0.7803 | 0.7563 | 0.6717 | 3740.6 |
| 3 | per-channel | 28.43 | 0.4377 | 0.6730 | 0.7459 | 0.6598 | 1749.9 |

Total elapsed time (including calibration data preparation): 13504.3 s (~225 min).

### JointQ (λ=0.0, actorder=true)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| — (Original) | — | 9.72 | 0.5657 | 0.8093 | 0.7775 | 0.6756 | — |
| 4 | 128 | NaN* | 0.2270 | 0.2508 | 0.4951 | 0.4957 | 2263.7 |
| 4 | per-channel | 10.45 | 0.5614 | 0.7959 | 0.7693 | 0.6946 | 1341.3 |
| 3 | 128 | NaN* | 0.2270 | 0.2508 | 0.4951 | 0.4957 | 2546.9 |
| 3 | per-channel | 26.72 | 0.4462 | 0.6772 | 0.7481 | 0.6622 | 1756.7 |

Total elapsed time (including calibration data preparation): 10883.6 s (~181 min).

\* `group_size=128` variants produced NaN perplexity and chance-level accuracy, indicating a failed quantization. Per-channel variants in the same run are unaffected.

## Environment

- GPU: NVIDIA B200 × 1

## Output

Results are saved under the `output_dir` directory (default: `qwen3-8b/`):

- `quantization_statistics_<quantizer_name>.json` — per-layer quantization error statistics
- Perplexity and accuracy results are printed to stdout

## Notes

This benchmark internally uses `Runner.quantize_with_calibration_chunked`, which can run multiple quantizers simultaneously without QEP. However, it requires the entire model to fit on the GPU and involves redundant forward passes. Addressing these limitations is future work.
