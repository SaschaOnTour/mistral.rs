# Phase B0 Benchmark Results

Model: Qwen/Qwen3-0.6B
Decode tokens: 128
Date: 2026-04-05T00:08:32+00:00
Hardware: AMD EPYC 7H12 64-Core Processor
GPU: NVIDIA GeForce RTX 3090


## Context: 512 tokens

| Variant | Total Time | Prefill tok/s | Decode tok/s | Decode ms/T | Peak VRAM |
|---------|-----------|---------------|-------------|-------------|-----------|
| CPU Normal | 64s | 144.2 ± 1.7 | 8.5 ± 0.1 | 117.49 ms/T | 1 MiB |
| CPU TQ3 | 95s | 140.3 ± 4.7 | 5.2 ± 0.0 | 194.01 ms/T | 262 MiB |
| GPU Normal | 6s | 18574.8 ± 1106.3 | 161.8 ± 0.3 | 6.18 ms/T | 21792 MiB |
| GPU TQ3 | 19s | 4034.6 ± 530.4 | 26.8 ± 0.2 | 37.32 ms/T | 1728 MiB |

## Context: 2048 tokens

| Variant | Total Time | Prefill tok/s | Decode tok/s | Decode ms/T | Peak VRAM |
|---------|-----------|---------------|-------------|-------------|-----------|
