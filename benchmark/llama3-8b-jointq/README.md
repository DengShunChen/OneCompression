# Llama-3-8B JointQ Benchmark

JointQ benchmark for [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) using OneComp v0.3.7.

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
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B

# actorder (λ=0.2)
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B \
    jointq.actorder=true output_dir=llama3-8b-actorder

# noreg (λ=0.0)
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B \
    jointq.regularization_lambda=0.0 output_dir=llama3-8b-noreg

# noreg+actorder (λ=0.0)
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B \
    jointq.regularization_lambda=0.0 jointq.actorder=true output_dir=llama3-8b-noreg-actorder
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
| — (Original) | — | 6.14 | 0.5401 | 0.7761 | 0.8063 | 0.7380 | — |
| 4 | 128 | 6.67 | 0.5196 | 0.7837 | 0.7954 | 0.7230 | 1196.8 |
| 4 | per-channel | 8.46 | 0.4753 | 0.7277 | 0.7726 | 0.7269 | 3110.7 |
| 3 | 128 | 9.26 | 0.4454 | 0.6831 | 0.7644 | 0.7064 | 1895.3 |
| 3 | per-channel | 21.15 | 0.3567 | 0.5707 | 0.7089 | 0.6946 | 2946.2 |

Total elapsed time (including calibration data preparation): 11781.8 s (~196 min).

### JointQ (λ=0.2, actorder=true)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| — (Original) | — | 6.14 | 0.5401 | 0.7761 | 0.8063 | 0.7380 | — |
| 4 | 128 | 6.66 | 0.5145 | 0.7786 | 0.7943 | 0.7182 | 1165.3 |
| 4 | per-channel | 8.44 | 0.4735 | 0.7302 | 0.7764 | 0.7245 | 3094.8 |
| 3 | 128 | 11.41 | 0.4710 | 0.7155 | 0.7780 | 0.7056 | 1866.9 |
| 3 | per-channel | 20.86 | 0.3567 | 0.5694 | 0.7111 | 0.6875 | 2917.6 |

Total elapsed time (including calibration data preparation): 11646.3 s (~194 min).

### JointQ (λ=0.0, actorder=false)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| — (Original) | — | 6.14 | 0.5401 | 0.7761 | 0.8063 | 0.7380 | — |
| 4 | 128 | 6.67 | 0.5239 | 0.7786 | 0.7960 | 0.7316 | 1384.5 |
| 4 | per-channel | 7.55 | 0.4974 | 0.7428 | 0.7878 | 0.7103 | 873.5 |
| 3 | 128 | 8.89 | 0.4462 | 0.7100 | 0.7709 | 0.7277 | 1684.9 |
| 3 | per-channel | 307.00 | 0.3695 | 0.6246 | 0.7378 | 0.6851 | 1668.3 |

Total elapsed time (including calibration data preparation): 8184.8 s (~136 min).

### JointQ (λ=0.0, actorder=true)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| — (Original) | — | 6.14 | 0.5401 | 0.7761 | 0.8063 | 0.7380 | — |
| 4 | 128 | 6.61 | 0.5239 | 0.7837 | 0.7922 | 0.7316 | 1174.2 |
| 4 | per-channel | 7.48 | 0.4889 | 0.7744 | 0.7856 | 0.7222 | 1038.8 |
| 3 | 128 | 10.47 | 0.4590 | 0.7168 | 0.7666 | 0.7088 | 1548.9 |
| 3 | per-channel | 118.56 | 0.3549 | 0.6065 | 0.7176 | 0.6772 | 1730.7 |

Total elapsed time (including calibration data preparation): 8061.7 s (~134 min).

## Environment

- GPU: NVIDIA B200 × 1

## Output

Results are saved under the `output_dir` directory (default: `llama3-8b/`):

- `quantization_statistics_<quantizer_name>.json` — per-layer quantization error statistics
- Perplexity and accuracy results are printed to stdout

## Notes

This benchmark internally uses `Runner.quantize_with_calibration_chunked`, which can run multiple quantizers simultaneously without QEP. However, it requires the entire model to fit on the GPU and involves redundant forward passes. Addressing these limitations is future work.
