# Appendix: Detailed Serving Breakdowns, Sweeps, and Output Verification (MiniMax-H3 on TPU)

This document preserves the complete serving optimizations, workload specifications, cold-compile times, device-synchronized stage breakdowns, parameter sweeps, multi-host concurrency results, TPU `v5p` runs, and `ffprobe` output verification for `MiniMax-H3` on Google Cloud TPU `v7x` and `v5p`. For headline results and deployment commands, see [`README.md`](./README.md).

---

## 1. TorchTPU Serving Optimizations & Methodology

[`scripts/run_minimax_h3_tpu_benchmark.py`](./scripts/run_minimax_h3_tpu_benchmark.py) and `sglang-torchtpu` combine seven TPU-specific serving optimizations:

1. **Ulysses-8 Sequence Parallelism (`--num-gpus=8 --ulysses-degree=8` on `v7x`, `--ulysses-degree=4` on `v5p`):** Shards MiniMax-H3's 56 attention heads cleanly across all 8 `v7x` TensorCores (`7` heads × `128` dim per core) or 4 `v5p` chips (`14` heads per chip).
2. **Folded Qwen3-VL Multimodal Encoder Parallelism (`--encoder-parallel=fold`):** Shards the `66.7 GB` multimodal text/vision encoder across all 8 TensorCores (`8.34 GB/core` on `v7x`) or 4 chips (`16.7 GB/chip` on `v5p`).
3. **Pallas Splash Attention (`--attention-backend=tpu_splash_attn`):** Runs dense bidirectional FlashAttention (`cache_dit` off) across variable-length packed `[text, audio, video]` and reference streams, paired with static XLA scatter ops in Qwen3-VL for arbitrary multi-reference combinations (`1` video + `5` images).
4. **Online FP8 DiT Quantization + Pre-Quantization LoRA Fusion:** Merges all 312 `lightx2v/Minimax-h3-Turbo` LoRA adapter pairs in `bfloat16` (handling the SwiGLU `[value; gate] -> [gate; value]` channel permutation) before quantizing DiT linear weights to native `v7x` `fp8` MXU (`torch._scaled_mm(..., out_dtype=torch.bfloat16)`), while keeping attention and normalizations in `bfloat16`.
5. **Selective Encoder/VAE Offload:** Offloads Qwen3-VL and VAE weights between stages when needed so extreme 6-reference `1792×768 × 345`-frame sequences fit inside `70.2 GB` per `96 GiB` `v7x` TensorCore.
6. **Disaggregated CPU MP4 Encoding ([`scripts/video_encode_server.py`](./scripts/video_encode_server.py)):** Streams raw decoded RGB24 frames and `float32` 32 kHz stereo PCM audio over HTTP to dedicated `c3-standard-44` Sapphire Rapids CPU pods running multi-threaded `libx264` + AAC muxing in `/dev/shm`.

### Metric Definitions
- **`inference_time_s`:** Server-side net inference time reported by SGLang from pipeline start through decoded RGB video frames and stereo audio samples (excluding HTTP transport, queueing, input download, and MP4 muxing).
- **End-to-End (`e2e_s`):** Client-observed wall time including request submission, polling, and disaggregated CPU MP4 muxing (`Encode + I/O = End-to-End - inference_time_s`).
- **First Request:** Cold end-to-end latency on a fresh server for a new shape, including XLA compilation. Warm runs repeat the identical request (`concurrency = 1`) and report the median.
- **Steps vs. DiT Passes:** For `num_inference_steps = N`, the MiniMax-H3 scheduler evaluates the DiT `N - 1` times (`4` steps = `3` passes; `5` steps = `4` passes; `26` steps = `25` passes; `50` steps = `49` passes).

---

## 2. Exact Workload Specifications & Input Checksums

All workloads run with stereo audio generation enabled, step caching disabled (`cache_dit` off), dense attention, default flow shift (`12.0` video / `3.0` audio unless noted), and H.264 (`24 fps`) + stereo AAC (`32 kHz`) output.

1. **Workload #1 — Ref2VA, 6 References (`1792×768` `21:9`, `345` frames / `14.375 s`, `4` steps, seed `0`):**
   - Weights: `MiniMaxAI/MiniMax-H3` `Ref2VA` + `lightx2v/Minimax-h3-Turbo` `minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors` (`alpha=8, scale=1.0`, SHA-256 `9e642fc8749c74f8da5e2382877ab5c7aa37b9a73b7fd0d6d457bd1b3cb1ae99`), merged before online FP8.
   - Exact prompt (preserving the two spaces after `Viking,`):
     `Using Image One the video make sound clip music in English translation , testosterone Viking,  use images attached to make the image identify of male`
   - Reference inputs (in order) & SHA-256 checksums:
     - `h3_r2v_compare_ref_14s.mp4`: `f938ce2c07c6e67e04fd6bcb53bdfef1f30746267f54d5beb1f5a6763e34703f`
     - `h3_r2v_compare_img_01.jpg`: `a7357a220c33abbab6d699204163746e531036c3792a7ad1b02586a573270696`
     - `h3_r2v_compare_img_02.png`: `8d2bb40e2d27af1f0be3b5b428180be62f006e2413916fb2f9eff2d12c7d6ee7`
     - `h3_r2v_compare_img_03.jpg`: `60a01bf8b775bdffe00464387fc6cc659e6155aa0c4c59defcf67431d14194a9`
     - `h3_r2v_compare_img_04.jpg`: `6f16ef40954fa0285c6ddfe0540df9f66a18837137073d38e52f94d2a7379fc9`
     - `h3_r2v_compare_img_05.png`: `3f6cd9f09467b27bfcbefdac143bece1413e0ab8205c13ce46d7fcefb79b1e40`
2. **Workloads #2–#4 — I2V (`1344×768` `16:9`, `345` frames / `14.375 s`, `4` steps, seed `0`):**
   - Input image: `gpt_dragon_1080p.jpg` (SHA-256 `fa2bd7a388a2e3d3951ec2d531dee127a9a07f8db53cd00a22341bc689d05bfc`).
   - Prompt: `The camera slowly pushes in with epic bgm. Keep the subject identity.`
   - Weights: #2 `FL2VA` base (`fp8`); #3 `FL2VA` + `minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors` (`alpha=8, scale=1.0`, `fp8`); #4 `FL2VA` base (`bf16`).
3. **Workloads #5–#6 — T2VA (`1344×768` `16:9`, `124` frames / `5.17 s`, `50` steps, seed `1101`):**
   - Prompt: `At night, while their owner sleeps in a bedroom, three cats march in loudly playing tiny brass instruments, then abruptly file out.`
   - Weights: `FL2VA` base in online `fp8` (#5) and lossless `bf16` (#6).
4. **Workloads #7–#8 — Ref2VA, 1 Image Reference (`1344×768` `16:9`, `124` frames / `5.17 s`, `50` steps, seed `1101`):**
   - Same prompt as #5–#6 + reference `h3_r2v_compare_img_01.jpg`; `Ref2VA` base in online `fp8` (#7) and lossless `bf16` (#8).

---

## 3. Complete Single-Host TPU `v7x` Results (Including Cold Compile & Peak Memory)

| # | Weights & Precision | Warm Runs (`inference_time_s`) | Median `inference_time_s` | Median End-to-End | Encode + I/O | Cold First Request | Peak HBM / Core (MB) | Raw Record | Video |
| :-: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| 1 | Ref2VA + Turbo `ref2v v0.1`, FP8 | 277.135 / 277.635 / 278.218 | **277.635 s** | 283.891 s | 6.256 s | 1,949.16 s | 70,207 | [jsonl](./results/v7x_1host/ref2va_6refs_4steps_fp8.jsonl) | [MP4](./videos/v7x_ref2va_6refs_1792x768_345f_4steps_fp8_turbo-ref2v-v0.1.mp4) |
| 1p | Same, with sync stage profiling | 277.466 / 277.780 | 277.623 s | 284.052 s | 6.429 s | 1,970.24 s | 70,752 | [jsonl](./results/v7x_1host/ref2va_6refs_4steps_fp8_stage_profile.jsonl) | — |
| 2 | FL2VA base, FP8 | 67.480 / 68.100 / 68.115 | **68.100 s** | 72.887 s | 4.787 s | 445.88 s | 64,322 | [jsonl](./results/v7x_1host/i2v_4steps_fp8.jsonl) | [MP4](./videos/v7x_i2v_1344x768_345f_4steps_fp8_base.mp4) |
| 2p | Same, with sync stage profiling | 68.169 / 67.924 | 68.046 s | 72.882 s | 4.836 s | 466.27 s | 64,297 | [jsonl](./results/v7x_1host/i2v_t2va_fp8_stage_profile.jsonl) | — |
| 3 | FL2VA + Turbo `fl2v v1.2 768p`, FP8 | 67.820 / 67.683 / 67.468 | **67.683 s** | 72.345 s | 4.662 s | 76.88 s* | 64,322 | [jsonl](./results/v7x_1host/i2v_4steps_fp8.jsonl) | [MP4](./videos/v7x_i2v_1344x768_345f_4steps_fp8_turbo-fl2v-v1.2.mp4) |
| 4 | FL2VA base, BF16 | 67.990 / 68.043 / 67.824 | **67.990 s** | 72.881 s | 4.891 s | 498.53 s | 73,834 | [jsonl](./results/v7x_1host/i2v_4steps_bf16.jsonl) | [MP4](./videos/v7x_i2v_1344x768_345f_4steps_bf16_base.mp4) |
| 5 | FL2VA base, FP8 (stage-profiled) | 122.067 / 121.986 | **122.026 s** | 123.837 s | 1.811 s | —* | 48,690 | [jsonl](./results/v7x_1host/i2v_t2va_fp8_stage_profile.jsonl) | [MP4](./videos/v7x_t2va_1344x768_124f_50steps_fp8_base.mp4) |
| 6 | FL2VA base, BF16 (lossless) | 132.236 (1 warm run) | **132.236 s** | 134.676 s | 2.440 s | 564.23 s | 65,958 | [jsonl](./results/v7x_1host/t2va_50steps_bf16.jsonl) | [MP4](./videos/v7x_t2va_1344x768_124f_50steps_bf16_base.mp4) |
| 7 | Ref2VA base (1 ref), FP8 | 205.969 / 206.438 / 206.341 | **206.341 s** | 208.533 s | 2.192 s | 386.02 s | 50,425 | [jsonl](./results/v7x_1host/ref2va_1ref_50steps_fp8.jsonl) | [MP4](./videos/v7x_ref2va_1ref_1344x768_124f_50steps_fp8_base.mp4) |
| 8 | Ref2VA base (1 ref), BF16 (lossless) | 213.326 / 216.078 / 215.562 | **215.562 s** | 217.961 s | 2.399 s | 302.39 s* | 57,406 | [jsonl](./results/v7x_1host/ref2va_1ref_50steps_bf16.jsonl) | [MP4](./videos/v7x_ref2va_1ref_1344x768_124f_50steps_bf16_base.mp4) |

*\*Compilation cache notes:* Workload #3 reused the XLA compilation cache from #2 (`76.88 s` first request); #5 ran on the same server after the 5-step `124`-frame T2VA profile run had already compiled that shape; #8 reused partial cache entries from #7. Enabling `SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1` changed I2V latency by `<0.1%` (`68.046 s` vs. `68.100 s`).

---

## 4. Device-Synchronized Stage-Level Latency Breakdown (`1` Host `v7x`, Online `fp8`)

Measured with `SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1` (mean of 2 warm requests). Raw JSON traces are in [`results/perf/v7x_fl2va_t2va_fp8/`](./results/perf/v7x_fl2va_t2va_fp8/) and [`results/perf/v7x_ref2va_6refs_fp8/`](./results/perf/v7x_ref2va_6refs_fp8/).

| Workload | Steps (`DiT Passes`) | Text Encode | Visual Encode | Audio Encode | DiT Denoising (`Mean / Pass`) | Video + Audio Decode | Other (Offload / Prep) | Total |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| #1 Ref2VA, 6 refs, `1792×768`, `345` frames | 4 (`3`) | 3.88 s | 24.89 s | 2.04 s | 219.25 s (`72.41 s`) | 22.62 s | 4.94 s | **277.62 s** |
| #2 I2V, `1344×768`, `345` frames | 4 (`3`) | 2.72 s | 2.63 s | — | 37.71 s (`12.53 s`) | 19.67 s | 5.32 s | **68.05 s** |
| I2V, `1344×768`, `345` frames | 5 (`4`) | 2.76 s | 2.69 s | — | 50.33 s (`12.55 s`) | 19.44 s | 5.19 s | **80.40 s** |
| T2VA, `1344×768`, `124` frames | 5 (`4`) | 2.65 s | — | — | 8.78 s (`2.18 s`) | 9.82 s | 7.36 s | **28.60 s** |
| #5 T2VA, `1344×768`, `124` frames | 50 (`49`) | 2.64 s | — | — | 102.29 s (`2.09 s`) | 9.66 s | 7.44 s | **122.03 s** |

---

## 5. Single-Host Parameter Sweeps on TPU `v7x`

### A. Step Count, Precision, and Flow Shift (`I2V`, `1344×768`, `345` Frames)

| Steps (`DiT Passes`) | Weights | Precision | Flow Shift | Runs | Median `inference_time_s` | Median End-to-End | Raw Record | Video |
| :---: | :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| 4 (`3`) | FL2VA base | FP8 | 12.0 | 3 | 68.100 s | 72.887 s | [jsonl](./results/v7x_1host/i2v_4steps_fp8.jsonl) | [MP4](./videos/v7x_i2v_1344x768_345f_4steps_fp8_base.mp4) |
| 4 (`3`) | Turbo `fl2v v1.2` | FP8 | 12.0 | 3 | **67.683 s** | 72.345 s | [jsonl](./results/v7x_1host/i2v_4steps_fp8.jsonl) | [MP4](./videos/v7x_i2v_1344x768_345f_4steps_fp8_turbo-fl2v-v1.2.mp4) |
| 4 (`3`) | Turbo `fl2v v1.2` | FP8 | 6.0 | 2 | 67.950 s | 72.668 s | [jsonl](./results/v7x_1host/sweeps_fp8.jsonl) | [MP4](./videos/v7x_i2v_1344x768_345f_4steps_fp8_turbo-fl2v-v1.2_shift6.mp4) |
| 4 (`3`) | FL2VA base | BF16 | 12.0 | 3 | 67.990 s | 72.881 s | [jsonl](./results/v7x_1host/i2v_4steps_bf16.jsonl) | [MP4](./videos/v7x_i2v_1344x768_345f_4steps_bf16_base.mp4) |
| 5 (`4`) | FL2VA base | FP8 | 12.0 | 2 | 80.404 s | 85.222 s | [jsonl](./results/v7x_1host/i2v_t2va_fp8_stage_profile.jsonl) | — |
| 5 (`4`) | Turbo `fl2v v1.2` | FP8 | 12.0 | 2 | 80.545 s | 85.259 s | [jsonl](./results/v7x_1host/sweeps_fp8.jsonl) | [MP4](./videos/v7x_i2v_1344x768_345f_5steps_fp8_turbo-fl2v-v1.2.mp4) |
| 5 (`4`) | Turbo `fl2v v1.2` | FP8 | 6.0 | 3 | **80.345 s** | 84.886 s | [jsonl](./results/v7x_1host/sweeps_fp8.jsonl) | [MP4](./videos/v7x_i2v_1344x768_345f_5steps_fp8_turbo-fl2v-v1.2_shift6.mp4) |
| 5 (`4`) | Turbo `fl2v v1.2` | BF16 | 6.0 | 3 | 82.022 s | 86.396 s | [jsonl](./results/v7x_1host/sweeps_bf16.jsonl) | [MP4](./videos/v7x_i2v_1344x768_345f_5steps_bf16_turbo-fl2v-v1.2_shift6.mp4) |
| 26 (`25`) | FL2VA base | FP8 | 12.0 | 1 | **353.289 s** | 359.347 s | [jsonl](./results/v7x_1host/sweeps_fp8.jsonl) | [MP4](./videos/v7x_i2v_1344x768_345f_26steps_fp8_base.mp4) |
| 26 (`25`) | FL2VA base | BF16 | 12.0 | 1 | 382.555 s | 387.267 s | [jsonl](./results/v7x_1host/sweeps_bf16.jsonl) | [MP4](./videos/v7x_i2v_1344x768_345f_26steps_bf16_base.mp4) |

*(At 26 steps on `345`-frame `1344×768` I2V, online FP8 is `8.3%` faster than BF16: `353.29 s` vs. `382.56 s`.)*

### B. Video Length Sweep (`I2V`, `1344×768`, Turbo `fl2v v1.2`, 5 Steps, Flow Shift `6.0`)

| Frames (`24 fps`) | Clip Duration | Precision | Runs | Median `inference_time_s` | Median End-to-End | Raw Record | Video |
| ---: | ---: | :--- | ---: | ---: | ---: | :--- | :--- |
| 124 | 5.17 s | FP8 | 3 | 31.317 s | 33.346 s | [jsonl](./results/v7x_1host/sweeps_fp8.jsonl) | [MP4](./videos/v7x_i2v_1344x768_124f_5steps_fp8_turbo-fl2v-v1.2_shift6.mp4) |
| 124 | 5.17 s | BF16 | 3 | **30.622 s** | 32.831 s | [jsonl](./results/v7x_1host/sweeps_bf16.jsonl) | [MP4](./videos/v7x_i2v_1344x768_124f_5steps_bf16_turbo-fl2v-v1.2_shift6.mp4) |
| 243 | 10.13 s | FP8 | 2 | 53.521 s | 56.992 s | [jsonl](./results/v7x_1host/sweeps_fp8.jsonl) | — |
| 345 | 14.375 s | FP8 | 3 | 80.345 s | 84.886 s | [jsonl](./results/v7x_1host/sweeps_fp8.jsonl) | [MP4](./videos/v7x_i2v_1344x768_345f_5steps_fp8_turbo-fl2v-v1.2_shift6.mp4) |
| 362 | 15.08 s | FP8 | 2 | 91.675 s | 96.809 s | [jsonl](./results/v7x_1host/sweeps_fp8.jsonl) | — |

### C. `FastVideo/FastH3` Distilled Checkpoint (`T2VA`, `1344×768`, 5 Steps / 4 DiT Passes, Online `fp8`)

| Frames (`24 fps`) | Clip Duration | Runs | Median `inference_time_s` | Median End-to-End | Cold First Request | Raw Record | Video |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| 124 | 5.17 s | 3 | **29.043 s** | 30.756 s | 36.25 s | [jsonl](./results/v7x_1host/fasth3_t2va_fp8.jsonl) | [MP4](./videos/v7x_fasth3_t2va_1344x768_124f_5steps_fp8.mp4) |
| 345 | 14.375 s | 2 | **76.783 s** | 81.198 s | 432.70 s | [jsonl](./results/v7x_1host/fasth3_t2va_fp8.jsonl) | [MP4](./videos/v7x_fasth3_t2va_1344x768_345f_5steps_fp8.mp4) |
| 362 | 15.08 s | 2 | **80.966 s** | 85.352 s | 469.01 s | [jsonl](./results/v7x_1host/fasth3_t2va_fp8.jsonl) | — |

---

## 6. Full Multi-Host Throughput Sweep on TPU `v7x`

Workload: `I2V` (`1344×768`), `FL2VA` + merged Turbo `fl2v v1.2`, online `fp8`, `5` steps (`4` DiT passes).

| Topology | Chips (`Replicas`) | Output | Concurrency | Total Requests | Wall Time | Videos / Min | Video-s / Wall-s | `p50` End-to-End | `p90` End-to-End | Median `inference_time_s` | Raw Record |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 2 hosts (`v7x-8`) | 8 (`2`) | 345 frames (`14.375 s`) | 2 | 4 | 172.71 s | 1.390 | 0.333 | 85.291 s | 87.889 s | 80.265 s | [jsonl](./results/v7x_2host/dp2_throughput_fp8.jsonl) |
| 2 hosts (`v7x-8`) | 8 (`2`) | 124 frames (`5.17 s`) | 4 | 8 | 195.49 s | 2.455 | 0.211 | 96.543 s | 98.260 s | 30.373 s | [jsonl](./results/v7x_2host/dp2_throughput_fp8.jsonl) |
| 4 hosts (`v7x-16`) | 16 (`4`) | 345 frames (`14.375 s`) | 4 | 8 | 196.81 s | **2.439** | **0.584** | 93.373 s | 100.914 s | 88.843 s | [jsonl](./results/v7x_4host/dp4_throughput_fp8.jsonl) |
| 4 hosts (`v7x-16`) | 16 (`4`) | 345 frames (`14.375 s`) | 8 | 12 | 304.11 s | 2.368 | 0.567 | 175.799 s | 202.598 s | 89.799 s | [jsonl](./results/v7x_4host/dp4_throughput_fp8.jsonl) |
| 4 hosts (`v7x-16`) | 16 (`4`) | 124 frames (`5.17 s`) | 4 | 12 | 140.91 s | **5.110** | **0.440** | 40.989 s | 49.327 s | 38.741 s | [jsonl](./results/v7x_4host/dp4_throughput_fp8.jsonl) |
| 4 hosts (`v7x-16`) | 16 (`4`) | 124 frames (`5.17 s`) | 8 | 16 | 188.20 s | 5.101 | 0.439 | 72.936 s | 99.101 s | 40.247 s | [jsonl](./results/v7x_4host/dp4_throughput_fp8.jsonl) |

*(Increasing concurrency beyond the replica count `c = 8` on 4 replicas keeps throughput saturated at `2.37–2.44` / `5.10–5.11` videos/min while adding request queueing latency.)*

---

## 7. Complete Single-Host TPU `v5p` Results (`4` Chips, `bf16`, Ulysses-4 + SDPA)

On a single 4-chip `v5p` host (`380 GiB` total HBM), `bf16` inference runs Ulysses-4 with `--encoder-parallel=fold`, SDPA attention, encoder/VAE offload, and FSDP weight sharding on 50-step T2VA. Multi-reference 6-input `Ref2VA` (`1792×768 × 345` frames) was not completed on a single 4-chip `v5p` host.

| Workload & Output | Steps (`DiT Passes`) | Weights (`bf16`) | Warm Runs (`s`) | Median `inference_time_s` | Median End-to-End | Cold First Request | Peak HBM / Chip (MB) | Raw Record | Video |
| :--- | :---: | :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| I2V (`1344×768`, `345` frames) | 4 (`3`) | FL2VA base | 153.510 / 152.794 / 160.661 | 153.510 s | 159.095 s | 1,687.56 s | 73,307 | [jsonl](./results/v5p_1host/i2v_4steps_bf16.jsonl) | [MP4](./videos/v5p_i2v_1344x768_345f_4steps_bf16_base.mp4) |
| I2V (`1344×768`, `345` frames) | 4 (`3`) | FL2VA + Turbo `fl2v v1.2` | 143.815 / 147.202 / 146.113 | **146.113 s** | 150.710 s | 1,635.83 s | 73,307 | [jsonl](./results/v5p_1host/i2v_4steps_bf16.jsonl) | [MP4](./videos/v5p_i2v_1344x768_345f_4steps_bf16_turbo-fl2v-v1.2.mp4) |
| I2V (`1344×768`, `345` frames, shift `6.0`) | 5 (`4`) | FL2VA + Turbo `fl2v v1.2` | 3 warm runs | 175.179 s | 180.335 s | 1,643.94 s | 73,307 | [jsonl](./results/v5p_1host/i2v_5steps_bf16.jsonl) | [MP4](./videos/v5p_i2v_1344x768_345f_5steps_bf16_turbo-fl2v-v1.2_shift6.mp4) |
| T2VA (`1344×768`, `124` frames) | 50 (`49`) | FL2VA base (FSDP) | 327.131 / 330.833 / 333.977 | 330.833 s | 332.936 s | 948.62 s | 32,430 | [jsonl](./results/v5p_1host/t2va_50steps_bf16.jsonl) | [MP4](./videos/v5p_t2va_1344x768_124f_50steps_bf16_base.mp4) |
| T2VA (`1344×768`, `124` frames) | 5 (`4`) | `FastVideo/FastH3` | 3 warm runs | **105.089 s** | 107.130 s | 1,015.31 s | — | [jsonl](./results/v5p_1host/fasth3_t2va_bf16.jsonl) | [MP4](./videos/v5p_fasth3_t2va_1344x768_124f_5steps_bf16.mp4) |

---

## 8. Output Verification (`ffprobe`)

| Generated Video File | Size (Bytes) | Duration | Video Stream | Audio Stream |
| :--- | ---: | ---: | :--- | :--- |
| [v5p_fasth3_t2va_1344x768_124f_5steps_bf16.mp4](./videos/v5p_fasth3_t2va_1344x768_124f_5steps_bf16.mp4) | 1,414,186 | 5.175 s | `h264` 1344×768, 24 fps, 124 frames | `aac` 32 kHz, 2 ch |
| [v5p_i2v_1344x768_345f_4steps_bf16_base.mp4](./videos/v5p_i2v_1344x768_345f_4steps_bf16_base.mp4) | 11,809,982 | 14.375 s | `h264` 1344×768, 24 fps, 345 frames | `aac` 32 kHz, 2 ch |
| [v5p_i2v_1344x768_345f_4steps_bf16_turbo-fl2v-v1.2.mp4](./videos/v5p_i2v_1344x768_345f_4steps_bf16_turbo-fl2v-v1.2.mp4) | 8,870,951 | 14.375 s | `h264` 1344×768, 24 fps, 345 frames | `aac` 32 kHz, 2 ch |
| [v5p_i2v_1344x768_345f_5steps_bf16_turbo-fl2v-v1.2_shift6.mp4](./videos/v5p_i2v_1344x768_345f_5steps_bf16_turbo-fl2v-v1.2_shift6.mp4) | 8,156,872 | 14.375 s | `h264` 1344×768, 24 fps, 345 frames | `aac` 32 kHz, 2 ch |
| [v5p_t2va_1344x768_124f_50steps_bf16_base.mp4](./videos/v5p_t2va_1344x768_124f_50steps_bf16_base.mp4) | 1,352,762 | 5.175 s | `h264` 1344×768, 24 fps, 124 frames | `aac` 32 kHz, 2 ch |
| [v7x_fasth3_t2va_1344x768_124f_5steps_fp8.mp4](./videos/v7x_fasth3_t2va_1344x768_124f_5steps_fp8.mp4) | 1,526,973 | 5.175 s | `h264` 1344×768, 24 fps, 124 frames | `aac` 32 kHz, 2 ch |
| [v7x_fasth3_t2va_1344x768_345f_5steps_fp8.mp4](./videos/v7x_fasth3_t2va_1344x768_345f_5steps_fp8.mp4) | 6,685,022 | 14.375 s | `h264` 1344×768, 24 fps, 345 frames | `aac` 32 kHz, 2 ch |
| [v7x_i2v_1344x768_124f_5steps_bf16_turbo-fl2v-v1.2_shift6.mp4](./videos/v7x_i2v_1344x768_124f_5steps_bf16_turbo-fl2v-v1.2_shift6.mp4) | 3,453,796 | 5.175 s | `h264` 1344×768, 24 fps, 124 frames | `aac` 32 kHz, 2 ch |
| [v7x_i2v_1344x768_124f_5steps_fp8_turbo-fl2v-v1.2_shift6.mp4](./videos/v7x_i2v_1344x768_124f_5steps_fp8_turbo-fl2v-v1.2_shift6.mp4) | 3,799,863 | 5.175 s | `h264` 1344×768, 24 fps, 124 frames | `aac` 32 kHz, 2 ch |
| [v7x_i2v_1344x768_345f_26steps_bf16_base.mp4](./videos/v7x_i2v_1344x768_345f_26steps_bf16_base.mp4) | 9,352,527 | 14.375 s | `h264` 1344×768, 24 fps, 345 frames | `aac` 32 kHz, 2 ch |
| [v7x_i2v_1344x768_345f_26steps_fp8_base.mp4](./videos/v7x_i2v_1344x768_345f_26steps_fp8_base.mp4) | 9,881,381 | 14.375 s | `h264` 1344×768, 24 fps, 345 frames | `aac` 32 kHz, 2 ch |
| [v7x_i2v_1344x768_345f_4steps_bf16_base.mp4](./videos/v7x_i2v_1344x768_345f_4steps_bf16_base.mp4) | 12,686,447 | 14.375 s | `h264` 1344×768, 24 fps, 345 frames | `aac` 32 kHz, 2 ch |
| [v7x_i2v_1344x768_345f_4steps_fp8_base.mp4](./videos/v7x_i2v_1344x768_345f_4steps_fp8_base.mp4) | 13,275,598 | 14.375 s | `h264` 1344×768, 24 fps, 345 frames | `aac` 32 kHz, 2 ch |
| [v7x_i2v_1344x768_345f_4steps_fp8_turbo-fl2v-v1.2.mp4](./videos/v7x_i2v_1344x768_345f_4steps_fp8_turbo-fl2v-v1.2.mp4) | 10,762,143 | 14.375 s | `h264` 1344×768, 24 fps, 345 frames | `aac` 32 kHz, 2 ch |
| [v7x_i2v_1344x768_345f_4steps_fp8_turbo-fl2v-v1.2_shift6.mp4](./videos/v7x_i2v_1344x768_345f_4steps_fp8_turbo-fl2v-v1.2_shift6.mp4) | 11,340,321 | 14.375 s | `h264` 1344×768, 24 fps, 345 frames | `aac` 32 kHz, 2 ch |
| [v7x_i2v_1344x768_345f_5steps_bf16_turbo-fl2v-v1.2_shift6.mp4](./videos/v7x_i2v_1344x768_345f_5steps_bf16_turbo-fl2v-v1.2_shift6.mp4) | 9,266,202 | 14.375 s | `h264` 1344×768, 24 fps, 345 frames | `aac` 32 kHz, 2 ch |
| [v7x_i2v_1344x768_345f_5steps_fp8_turbo-fl2v-v1.2.mp4](./videos/v7x_i2v_1344x768_345f_5steps_fp8_turbo-fl2v-v1.2.mp4) | 10,315,259 | 14.375 s | `h264` 1344×768, 24 fps, 345 frames | `aac` 32 kHz, 2 ch |
| [v7x_i2v_1344x768_345f_5steps_fp8_turbo-fl2v-v1.2_shift6.mp4](./videos/v7x_i2v_1344x768_345f_5steps_fp8_turbo-fl2v-v1.2_shift6.mp4) | 10,225,390 | 14.375 s | `h264` 1344×768, 24 fps, 345 frames | `aac` 32 kHz, 2 ch |
| [v7x_ref2va_1ref_1344x768_124f_50steps_bf16_base.mp4](./videos/v7x_ref2va_1ref_1344x768_124f_50steps_bf16_base.mp4) | 2,444,256 | 5.175 s | `h264` 1344×768, 24 fps, 124 frames | `aac` 32 kHz, 2 ch |
| [v7x_ref2va_1ref_1344x768_124f_50steps_fp8_base.mp4](./videos/v7x_ref2va_1ref_1344x768_124f_50steps_fp8_base.mp4) | 2,367,762 | 5.175 s | `h264` 1344×768, 24 fps, 124 frames | `aac` 32 kHz, 2 ch |
| [v7x_ref2va_6refs_1792x768_345f_4steps_fp8_turbo-ref2v-v0.1.mp4](./videos/v7x_ref2va_6refs_1792x768_345f_4steps_fp8_turbo-ref2v-v0.1.mp4) | 14,670,940 | 14.375 s | `h264` 1792×768, 24 fps, 345 frames | `aac` 32 kHz, 2 ch |
| [v7x_t2va_1344x768_124f_50steps_bf16_base.mp4](./videos/v7x_t2va_1344x768_124f_50steps_bf16_base.mp4) | 1,427,139 | 5.175 s | `h264` 1344×768, 24 fps, 124 frames | `aac` 32 kHz, 2 ch |
| [v7x_t2va_1344x768_124f_50steps_fp8_base.mp4](./videos/v7x_t2va_1344x768_124f_50steps_fp8_base.mp4) | 1,498,760 | 5.175 s | `h264` 1344×768, 24 fps, 124 frames | `aac` 32 kHz, 2 ch |
