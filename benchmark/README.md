# benchmark

Benchmark scripts for OneComp.
Configuration is managed with [Hydra](https://hydra.cc/).

> **Note:** Hydra is not a dependency of OneComp and must be installed separately.

## Installing Hydra

```bash
pip install hydra-core
```

Verify the installation:

```bash
python -c "import hydra; print(hydra.__version__)"
```

## Benchmarks

| Directory | Description |
|---|---|
| [llama3-8b-gptq/](llama3-8b-gptq/) | Llama-3-8B GPTQ (4bit/3bit × gs128/per-channel) |
| [llama3-8b-jointq/](llama3-8b-jointq/) | Llama-3-8B JointQ (4bit/3bit × gs128/per-channel) |
| [llama3-8b-qep-gptq/](llama3-8b-qep-gptq/) | Llama-3-8B QEP+GPTQ (4bit/3bit × gs128/per-channel) |
| [llama3-8b-various/](llama3-8b-various/) | Llama-3-8B Various quantizers with default parameters (no QEP) |
| [qwen3-8b-gptq/](qwen3-8b-gptq/) | Qwen3-8B GPTQ (4bit/3bit × gs128/per-channel) |
| [qwen3-8b-jointq/](qwen3-8b-jointq/) | Qwen3-8B JointQ (4bit/3bit × gs128/per-channel) |
| [qwen3-14b-gptq/](qwen3-14b-gptq/) | Qwen3-14B GPTQ (4bit/3bit × gs128/per-channel) |
| [qwen3-14b-jointq/](qwen3-14b-jointq/) | Qwen3-14B JointQ (4bit/3bit × gs128/per-channel) |

## Results Summary

Benchmark results using OneComp v0.3.7 on NVIDIA B200 × 1.

PPL = perplexity on WikiText-2 (↓ lower is better). Accuracy = 0-shot `acc_norm` where available, `acc` otherwise (winogrande) (↑ higher is better).

### Llama-3-8B: GPTQ vs JointQ

| Method | bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|---|
| Original | — | — | 6.14 | 0.5401 | 0.7761 | 0.8063 | 0.7380 | — |
| | | | | | | | | |
| GPTQ | 4 | 128 | 12.66 | 0.5026 | 0.7710 | 0.7922 | 0.7206 | **276.8** |
| JointQ | 4 | 128 | **6.67** | **0.5196** | **0.7837** | **0.7954** | **0.7230** | 1196.8 |
| | | | | | | | | |
| GPTQ | 4 | per-channel | 665.94 | 0.3089 | 0.5076 | 0.6861 | 0.6298 | **268.9** |
| JointQ | 4 | per-channel | **8.46** | **0.4753** | **0.7277** | **0.7726** | **0.7269** | 3110.7 |
| | | | | | | | | |
| GPTQ | 3 | 128 | 45.22 | 0.3097 | 0.4886 | 0.6610 | 0.6259 | **273.8** |
| JointQ | 3 | 128 | **9.26** | **0.4454** | **0.6831** | **0.7644** | **0.7064** | 1895.3 |
| | | | | | | | | |
| GPTQ | 3 | per-channel | 1721.06 | 0.2167 | 0.2862 | 0.5419 | 0.5004 | **268.4** |
| JointQ | 3 | per-channel | **21.15** | **0.3567** | **0.5707** | **0.7089** | **0.6946** | 2946.2 |

See [llama3-8b-gptq/](llama3-8b-gptq/) and [llama3-8b-jointq/](llama3-8b-jointq/) for full details.

### Llama-3-8B: GPTQ vs QEP+GPTQ

PPL = perplexity on WikiText-2 (↓ lower is better). Accuracy = 0-shot `acc_norm` where available, `acc` otherwise (winogrande) (↑ higher is better).

| Method | bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|---|
| Original | — | — | 6.14 | 0.5401 | 0.7761 | 0.8063 | 0.7380 | — |
| | | | | | | | | |
| GPTQ | 4 | 128 | 12.66 | 0.5026 | 0.7710 | 0.7922 | 0.7206 | **276.8** |
| QEP+GPTQ | 4 | 128 | **6.66** | **0.5265** | **0.7942** | **0.7916** | **0.7293** | 300.7 |
| | | | | | | | | |
| GPTQ | 4 | per-channel | 665.94 | 0.3089 | 0.5076 | 0.6861 | 0.6298 | **268.9** |
| QEP+GPTQ | 4 | per-channel | **7.67** | **0.4957** | **0.7542** | **0.7758** | **0.7269** | 297.7 |
| | | | | | | | | |
| GPTQ | 3 | 128 | 45.22 | 0.3097 | 0.4886 | 0.6610 | 0.6259 | **273.8** |
| QEP+GPTQ | 3 | 128 | **8.95** | **0.4352** | **0.6498** | **0.7546** | **0.6946** | 272.7 |
| | | | | | | | | |
| GPTQ | 3 | per-channel | 1721.06 | 0.2167 | 0.2862 | 0.5419 | 0.5004 | **268.4** |
| QEP+GPTQ | 3 | per-channel | **17.93** | **0.2688** | **0.4184** | **0.6806** | **0.6156** | 292.7 |

See [llama3-8b-gptq/](llama3-8b-gptq/) and [llama3-8b-qep-gptq/](llama3-8b-qep-gptq/) for full details.

### Qwen3-8B: GPTQ vs JointQ

PPL = perplexity on WikiText-2 (↓ lower is better). Accuracy = 0-shot `acc_norm` where available, `acc` otherwise (winogrande) (↑ higher is better).

| Method | bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|---|
| Original | — | — | 9.72 | 0.5657 | 0.8093 | 0.7775 | 0.6756 | — |
| | | | | | | | | |
| GPTQ | 4 | 128 | 10.29 | **0.5538** | **0.7946** | **0.7677** | 0.6709 | **275.8** |
| JointQ | 4 | 128 | 10.29 | 0.5469 | 0.7849 | 0.7650 | **0.6819** | 1183.8 |
| | | | | | | | | |
| GPTQ | 4 | per-channel | **10.97** | **0.5085** | 0.7412 | **0.7688** | **0.6693** | **269.0** |
| JointQ | 4 | per-channel | 11.33 | 0.4974 | **0.7483** | 0.7617 | 0.6330 | 2334.3 |
| | | | | | | | | |
| GPTQ | 3 | 128 | **11.71** | 0.4966 | 0.7273 | 0.7486 | 0.6440 | **275.2** |
| JointQ | 3 | 128 | 12.38 | 0.4966 | **0.7395** | **0.7568** | **0.6772** | 1895.9 |
| | | | | | | | | |
| GPTQ | 3 | per-channel | **20.21** | **0.3234** | 0.4293 | **0.6806** | **0.5501** | **269.1** |
| JointQ | 3 | per-channel | 42.10 | 0.2986 | **0.4987** | 0.6779 | 0.5478 | 2443.9 |

See [qwen3-8b-gptq/](qwen3-8b-gptq/) and [qwen3-8b-jointq/](qwen3-8b-jointq/) for full details.

### Qwen3-14B: GPTQ vs JointQ

PPL = perplexity on WikiText-2 (↓ lower is better). Accuracy = 0-shot `acc_norm` where available, `acc` otherwise (winogrande) (↑ higher is better).

| Method | bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|---|
| Original | — | — | 8.64 | 0.6032 | 0.8283 | 0.7971 | 0.7293 | — |
| | | | | | | | | |
| GPTQ | 4 | 128 | **8.85** | 0.5947 | 0.8182 | 0.7982 | 0.7316 | **481.9** |
| JointQ | 4 | 128 | 8.88 | **0.6007** | **0.8215** | **0.7992** | **0.7332** | 3024.7 |
| | | | | | | | | |
| GPTQ | 4 | per-channel | **9.15** | 0.5802 | 0.8056 | 0.7873 | 0.7056 | **470.1** |
| JointQ | 4 | per-channel | 9.96 | **0.5845** | **0.8110** | 0.7873 | **0.7080** | 5545.2 |
| | | | | | | | | |
| GPTQ | 3 | 128 | 10.10 | 0.5307 | 0.7727 | **0.7873** | 0.7001 | **480.1** |
| JointQ | 3 | 128 | **9.99** | **0.5401** | **0.7761** | 0.7791 | **0.7088** | 4936.1 |
| | | | | | | | | |
| GPTQ | 3 | per-channel | **13.72** | 0.3976 | 0.5745 | 0.7356 | 0.6172 | **468.4** |
| JointQ | 3 | per-channel | 23.55 | **0.4727** | **0.7247** | **0.7546** | **0.6504** | 5632.7 |

See [qwen3-14b-gptq/](qwen3-14b-gptq/) and [qwen3-14b-jointq/](qwen3-14b-jointq/) for full details.
