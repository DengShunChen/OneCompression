# Llama-3-8B Various Quantizers Benchmark

Benchmark for [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) using OneComp v0.3.7+jointq.
Various quantizers are run with their **default parameters** (no QEP).

## Quantizers

All 9 quantizers share calibration data accumulation (X^T X) for efficiency.

| Quantizer | Key Defaults |
|---|---|
| GPTQ | wbits=4, groupsize=-1 (per-channel), sym=True |
| JointQ | bits=4, group_size=128, symmetric=False |
| DBF | target_bits=1.5 |
| QUIP | wbits=4 |
| Onebit | iters=10 |
| RTN | wbits=4, groupsize=-1 (per-channel), sym=False |
| CQ | each_row=True |
| ARB | arb_iters=15, split_points=2 |
| QBB | wbits=4 |

## Benchmark Configuration

| Parameter | Value |
|---|---|
| num_calibration_samples | 1024 |
| calibration_strategy | drop_rand |
| max_length | 2048 |

### Evaluation

- Perplexity (WikiText-2)
- Accuracy (lm-eval-harness)

Both are computed for the original (unquantized) model and all quantized models.

## Usage

Requires [Hydra](https://hydra.cc/) (see [benchmark/README.md](../README.md) for installation).

Specify the path to the model via `model_path`:

```bash
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B
```

## Results

### Perplexity (WikiText-2, ↓ lower is better)

| Quantizer | PPL |
|---|---|
| Original | 6.14 |
| QUIP | 6.93 |
| JointQ | 6.58 |
| RTN | 8.52 |
| GPTQ | 651.13 |
| QBB | 215736.09 |
| CQ | 317645.78 |
| DBF | 357675.18 |
| ARB | 465093.00 |
| Onebit | 864565.38 |

### Accuracy (0-shot, ↑ higher is better)

Values are `acc_norm` where available, `acc` otherwise (winogrande).

| Quantizer | ARC-c | ARC-e | PIQA | WinoGrande |
|---|---|---|---|---|
| Original | 0.5401 | 0.7761 | 0.8063 | 0.7380 |
| QUIP | 0.5290 | 0.7858 | 0.7884 | 0.7395 |
| JointQ | 0.5230 | 0.7824 | 0.7895 | 0.7316 |
| RTN | 0.4872 | 0.7542 | 0.7628 | 0.7096 |
| GPTQ | 0.3114 | 0.5017 | 0.6861 | 0.6306 |
| CQ | 0.2722 | 0.2513 | 0.5092 | 0.5130 |
| ARB | 0.2671 | 0.2424 | 0.5141 | 0.5028 |
| QBB | 0.2577 | 0.2517 | 0.5316 | 0.4807 |
| DBF | 0.2568 | 0.2689 | 0.5332 | 0.4925 |
| Onebit | 0.2517 | 0.2555 | 0.5141 | 0.5185 |

### Quantization Time

| Quantizer | Time (s) |
|---|---|
| ARB | 42.1 |
| RTN | 52.8 |
| Onebit | 73.6 |
| GPTQ | 283.4 |
| QUIP | 405.8 |
| CQ | 587.0 |
| JointQ | 1034.3 |
| QBB | 3835.8 |
| DBF | 7575.2 |

Total elapsed time (including calibration data preparation and evaluation): 16558.4 s (~276 min).

## Environment

- GPU: NVIDIA B200 × 1
