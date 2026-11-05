# MiniMax-H3 Benchmark Results on Google Cloud TPU

- **Platforms:** Cloud TPU v7x (Ironwood, `tpu7x-standard-4t`: 4 chips, 8 TensorCores and 768 GiB HBM per host) and Cloud TPU v5p (`ct5p-hightpu-4t`: 4 chips per host), on Google Kubernetes Engine.
- **Serving stack:** `sglang-torchtpu` (TorchTPU 0.2.0, libtpu 0.0.50, JAX 0.11.2), dense Pallas Splash Attention, Ulysses sequence parallelism, `--encoder-parallel=fold`, plus a separate CPU MP4 encoding service.
- **Model:** `MiniMaxAI/MiniMax-H3`, Hugging Face revision `42ed227ee7df40d41602854ae760620d6eb651fe` (`FL2VA` and `Ref2VA` pipelines). Optional 4-step adapters come from `lightx2v/Minimax-h3-Turbo`. The distilled `FastVideo/FastH3` checkpoint was also tested.
- **Raw data:** every row links to its benchmark record in [`results/`](results/) and to its generated video in [`videos/`](videos/).

---

## 1. Summary: TPU v7x, one host (4 chips)

| # | Workload | Output | Weights and precision | Median `inference_time_s` | Median end-to-end |
| :-: | :--- | :--- | :--- | ---: | ---: |
| 1 | Ref2VA, 6 references, 15 s, 21:9, 4 steps | 1792×768, 345 frames | Ref2VA + Turbo `ref2v v0.1` (merged), online FP8 | **277.635 s** | 283.891 s |
| 2 | I2V, 15 s, 4 steps | 1344×768, 345 frames | FL2VA base, online FP8 | **68.100 s** | 72.887 s |
| 3 | I2V, 15 s, 4 steps | 1344×768, 345 frames | FL2VA + Turbo `fl2v v1.2 768p` (merged), online FP8 | **67.683 s** | 72.345 s |
| 4 | I2V, 15 s, 4 steps | 1344×768, 345 frames | FL2VA base, BF16 | **67.990 s** | 72.881 s |
| 5 | T2VA, 5 s, 50 steps | 1344×768, 124 frames | FL2VA base, online FP8 | **122.026 s** | 123.837 s |
| 6 | T2VA, 5 s, 50 steps | 1344×768, 124 frames | FL2VA base, BF16 (lossless) | **132.236 s** (1 run) | 134.676 s |
| 7 | Ref2VA, 1 image reference, 5 s, 50 steps | 1344×768, 124 frames | Ref2VA base, online FP8 | **206.341 s** | 208.533 s |
| 8 | Ref2VA, 1 image reference, 5 s, 50 steps | 1344×768, 124 frames | Ref2VA base, BF16 (lossless) | **215.562 s** | 217.961 s |

---

## 2. Workload specifications

All workloads run with:
- audio generation on and step caching off (`cache_dit` off);
- dense attention;
- the default flow shift (12.0 video / 3.0 audio) unless stated otherwise;
- concurrency 1 on a warm server.

Outputs are H.264 at 24 fps with 32 kHz stereo AAC.

### 2.1 Ref2VA: 6 references (workload #1)

| Parameter | Value |
| :--- | :--- |
| Task | Reference-to-video + audio (`Ref2VA` pipeline) |
| Duration / resolution / aspect ratio | 15 s / 768P / 21:9 |
| Output | 1792×768, 345 frames at 24 fps (14.375 s) |
| `num_inference_steps` / seed | 4 / 0 |
| References (in order) | 1 video, then 5 images (see the table below) |
| Weights | `MiniMaxAI/MiniMax-H3` Ref2VA + `lightx2v/Minimax-h3-Turbo` `minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors` (alpha 8, scale 1.0, SHA-256 `9e642fc8749c74f8da5e2382877ab5c7aa37b9a73b7fd0d6d457bd1b3cb1ae99`), merged into BF16 weights before online FP8 quantization |

Prompt (exact; note the two spaces after `Viking,`):

```text
Using Image One the video make sound clip music in English translation , testosterone Viking,  use images attached to make the image identify of male
```

| Reference | SHA-256 |
| :--- | :--- |
| `h3_r2v_compare_ref_14s.mp4` (video) | `f938ce2c07c6e67e04fd6bcb53bdfef1f30746267f54d5beb1f5a6763e34703f` |
| `h3_r2v_compare_img_01.jpg` | `a7357a220c33abbab6d699204163746e531036c3792a7ad1b02586a573270696` |
| `h3_r2v_compare_img_02.png` | `8d2bb40e2d27af1f0be3b5b428180be62f006e2413916fb2f9eff2d12c7d6ee7` |
| `h3_r2v_compare_img_03.jpg` | `60a01bf8b775bdffe00464387fc6cc659e6155aa0c4c59defcf67431d14194a9` |
| `h3_r2v_compare_img_04.jpg` | `6f16ef40954fa0285c6ddfe0540df9f66a18837137073d38e52f94d2a7379fc9` |
| `h3_r2v_compare_img_05.png` | `3f6cd9f09467b27bfcbefdac143bece1413e0ab8205c13ce46d7fcefb79b1e40` |

### 2.2 I2V: image-to-video + audio (workloads #2, #3 and #4)

| Parameter | Value |
| :--- | :--- |
| Task | Image-to-video + audio (`FL2VA` pipeline, first-frame conditioning) |
| Duration / resolution | 15 s / 768P (16:9 output for the 1920×1080 input image) |
| Output | 1344×768, 345 frames at 24 fps (14.375 s) |
| `num_inference_steps` / seed | 4 / 0 |
| Input image | `gpt_dragon_1080p.jpg`, SHA-256 `fa2bd7a388a2e3d3951ec2d531dee127a9a07f8db53cd00a22341bc689d05bfc` |
| Weights | #2: official FL2VA weights, no LoRA, online FP8. #3: FL2VA + `minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors` (alpha 8, scale 1.0), merged before online FP8. #4: official FL2VA weights, BF16/FP32, no quantization. |

Prompt:

```text
The camera slowly pushes in with epic bgm. Keep the subject identity.
```

### 2.3 T2VA: text-to-video + audio, 50 steps (workloads #5 and #6)

| Parameter | Value |
| :--- | :--- |
| Task | Text-to-video + audio (`FL2VA` pipeline, no conditioning) |
| Duration / resolution / aspect ratio | 5 s / 768P / 16:9 |
| Output | 1344×768, 124 frames at 24 fps (5.17 s) |
| `num_inference_steps` / seed / flow shift / audio flow shift | 50 / 1101 / 12.0 / 3.0 |
| Weights | Official FL2VA weights. #5: online FP8. #6: BF16/FP32, no quantization. |

Prompt:

```text
At night, while their owner sleeps in a bedroom, three cats march in loudly playing tiny brass instruments, then abruptly file out.
```

### 2.4 Ref2VA: 1 image reference, 50 steps (workloads #7 and #8)

| Parameter | Value |
| :--- | :--- |
| Task | Reference-to-video + audio (`Ref2VA` pipeline) |
| Duration / resolution / aspect ratio | 5 s / 768P / 16:9 |
| Output | 1344×768, 124 frames at 24 fps (5.17 s) |
| `num_inference_steps` / seed / flow shift / audio flow shift | 50 / 1101 / 12.0 / 3.0 |
| Reference | `h3_r2v_compare_img_01.jpg` (SHA-256 above) |
| Prompt | Same as §2.3 |
| Weights | Official Ref2VA weights. #7: online FP8. #8: BF16/FP32, no quantization. |

### 2.5 Measurement definitions

| Metric | Definition |
| :--- | :--- |
| `inference_time_s` | Server-side net inference time reported by SGLang for the request: pipeline start through decoded video frames and audio. Equals the total of the per-stage profile in §4. |
| End-to-end | Client-observed wall time: submission, polling, and MP4 muxing on the CPU encoding service. |
| Encode + I/O | End-to-end minus `inference_time_s`. |
| First request | End-to-end time of the first request for a new input shape on a fresh server. It includes XLA compilation. Later requests reuse the compiled program. |
| Steps vs. DiT passes | For `num_inference_steps = N`, the MiniMax-H3 scheduler evaluates the diffusion transformer N − 1 times (4 steps = 3 passes, 50 steps = 49 passes). The per-pass timings in §4 confirm this. |
| Peak memory | SGLang-reported peak device memory (`peak_memory_mb`) per TensorCore. Each v7x TensorCore has 96 GiB. |

---

## 3. Results by workload: TPU v7x, one host

`num_inference_steps` and the output shape for each workload are given in §2.

| # | Weights and precision | Warm runs (s) | Median `inference_time_s` | Median end-to-end | Encode + I/O | First request | Peak memory (MB) | Record | Video |
| :-: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| 1 | Ref2VA + Turbo `ref2v v0.1`, FP8 | 277.135 / 277.635 / 278.218 | **277.635 s** | 283.891 s | 6.256 s | 1,949.16 s | 70,207 | [jsonl](results/v7x_1host/ref2va_6refs_4steps_fp8.jsonl) | [MP4](videos/v7x_ref2va_6refs_1792x768_345f_4steps_fp8_turbo-ref2v-v0.1.mp4) |
| 1 | Same, with stage profiling (§4) | 277.466 / 277.780 | 277.623 s | 284.052 s | 6.429 s | 1,970.24 s | 70,752 | [jsonl](results/v7x_1host/ref2va_6refs_4steps_fp8_stage_profile.jsonl) | — |
| 2 | FL2VA base, FP8 | 67.480 / 68.100 / 68.115 | **68.100 s** | 72.887 s | 4.787 s | 445.88 s | 64,322 | [jsonl](results/v7x_1host/i2v_4steps_fp8.jsonl) | [MP4](videos/v7x_i2v_1344x768_345f_4steps_fp8_base.mp4) |
| 2 | Same, with stage profiling (§4) | 68.169 / 67.924 | 68.046 s | 72.882 s | 4.836 s | 466.27 s | 64,297 | [jsonl](results/v7x_1host/i2v_t2va_fp8_stage_profile.jsonl) | — |
| 3 | FL2VA + Turbo `fl2v v1.2 768p`, FP8 | 67.820 / 67.683 / 67.468 | **67.683 s** | 72.345 s | 4.662 s | 76.88 s | 64,322 | [jsonl](results/v7x_1host/i2v_4steps_fp8.jsonl) | [MP4](videos/v7x_i2v_1344x768_345f_4steps_fp8_turbo-fl2v-v1.2.mp4) |
| 4 | FL2VA base, BF16 | 67.990 / 68.043 / 67.824 | **67.990 s** | 72.881 s | 4.891 s | 498.531 s | 73,834 | [jsonl](results/v7x_1host/i2v_4steps_bf16.jsonl) | [MP4](videos/v7x_i2v_1344x768_345f_4steps_bf16_base.mp4) |
| 5 | FL2VA base, FP8, with stage profiling | 122.067 / 121.986 | **122.026 s** | 123.837 s | 1.811 s | — | 48,690 | [jsonl](results/v7x_1host/i2v_t2va_fp8_stage_profile.jsonl) | [MP4](videos/v7x_t2va_1344x768_124f_50steps_fp8_base.mp4) |
| 6 | FL2VA base, BF16 (lossless) | 132.236 (1 run) | **132.236 s** | 134.676 s | 2.440 s | 564.227 s | 65,958 | [jsonl](results/v7x_1host/t2va_50steps_bf16.jsonl) | [MP4](videos/v7x_t2va_1344x768_124f_50steps_bf16_base.mp4) |
| 7 | Ref2VA base, FP8 | 205.969 / 206.438 / 206.341 | **206.341 s** | 208.533 s | 2.192 s | 386.022 s | 50,425 | [jsonl](results/v7x_1host/ref2va_1ref_50steps_fp8.jsonl) | [MP4](videos/v7x_ref2va_1ref_1344x768_124f_50steps_fp8_base.mp4) |
| 8 | Ref2VA base, BF16 (lossless) | 213.326 / 216.078 / 215.562 | **215.562 s** | 217.961 s | 2.399 s | 302.388 s | 57,406 | [jsonl](results/v7x_1host/ref2va_1ref_50steps_bf16.jsonl) | [MP4](videos/v7x_ref2va_1ref_1344x768_124f_50steps_bf16_base.mp4) |

**Notes:**
- Workload #3's server reused the compilation cache from workload #2, so its first request is short.
- Workload #5 shares its server with the 5-step T2VA profile run in §4, which compiled the 124-frame shape first. That is why it has no separate first-request time.
- Workload #6 is a single warm request: the identical 50-step request, repeated once after the first request.
- Workload #8 ran after #7 on the same host and reused part of its compilation cache, so its first request is shorter.

---

## 4. Stage-level latency breakdown: TPU v7x, one host

These figures come from device-synchronized stage profiling (`SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1`). The TPU queue is synchronized at every stage and denoising-step boundary, so each stage's time is attributed correctly. Values are the mean of two warm requests, all in online FP8. Per-request profiles are in [`results/perf/`](results/perf/).

| Workload | Steps (DiT passes) | Text encode | Visual encode | Audio encode | DiT denoising (mean per pass) | Video + audio decode | Other | Total |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| #1 Ref2VA, 6 refs, 1792×768, 345 frames | 4 (3) | 3.88 s | 24.89 s | 2.04 s | 219.25 s (72.41 s) | 22.62 s | 4.94 s | **277.62 s** |
| #2 I2V, 1344×768, 345 frames | 4 (3) | 2.72 s | 2.63 s | — | 37.71 s (12.53 s) | 19.67 s | 5.32 s | **68.05 s** |
| I2V, 1344×768, 345 frames | 5 (4) | 2.76 s | 2.69 s | — | 50.33 s (12.55 s) | 19.44 s | 5.19 s | **80.40 s** |
| T2VA, 1344×768, 124 frames | 5 (4) | 2.65 s | — | — | 8.78 s (2.18 s) | 9.82 s | 7.36 s | **28.60 s** |
| #5 T2VA, 1344×768, 124 frames | 50 (49) | 2.64 s | — | — | 102.29 s (2.09 s) | 9.66 s | 7.44 s | **122.03 s** |

- **DiT denoising** is the full denoising stage: the transformer passes plus one-time stage setup (about 2.0 s for Ref2VA with 6 references, under 0.15 s otherwise).
- **Other** covers latent preparation plus host-side orchestration between stages, including moving the text encoder and VAE in and out of device memory.
- Per-pass DiT time depends on sequence length: about 2.1 s at 1344×768 with 124 frames, 12.5 s at 1344×768 with 345 frames, and 72.4 s at 1792×768 with 345 frames when 6 reference streams are packed into the joint context.

---

## 5. Parameter sweeps: TPU v7x, one host

### 5.1 Step count, precision and flow shift (I2V, 1344×768, 345 frames)

| `num_inference_steps` (DiT passes) | Weights | Precision | Flow shift | Runs | Median `inference_time_s` | Median end-to-end | Record | Video |
| :---: | :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| 4 (3) | FL2VA base | FP8 | 12.0 | 3 | **68.100 s** | 72.887 s | [jsonl](results/v7x_1host/i2v_4steps_fp8.jsonl) | [MP4](videos/v7x_i2v_1344x768_345f_4steps_fp8_base.mp4) |
| 4 (3) | Turbo `fl2v v1.2` | FP8 | 12.0 | 3 | **67.683 s** | 72.345 s | [jsonl](results/v7x_1host/i2v_4steps_fp8.jsonl) | [MP4](videos/v7x_i2v_1344x768_345f_4steps_fp8_turbo-fl2v-v1.2.mp4) |
| 4 (3) | Turbo `fl2v v1.2` | FP8 | 6.0 | 2 | **67.950 s** | 72.668 s | [jsonl](results/v7x_1host/sweeps_fp8.jsonl) | [MP4](videos/v7x_i2v_1344x768_345f_4steps_fp8_turbo-fl2v-v1.2_shift6.mp4) |
| 4 (3) | FL2VA base | BF16 | 12.0 | 3 | **67.990 s** | 72.881 s | [jsonl](results/v7x_1host/i2v_4steps_bf16.jsonl) | [MP4](videos/v7x_i2v_1344x768_345f_4steps_bf16_base.mp4) |
| 5 (4) | FL2VA base | FP8 | 12.0 | 2 | **80.404 s** | 85.222 s | [jsonl](results/v7x_1host/i2v_t2va_fp8_stage_profile.jsonl) | — |
| 5 (4) | Turbo `fl2v v1.2` | FP8 | 12.0 | 2 | **80.545 s** | 85.259 s | [jsonl](results/v7x_1host/sweeps_fp8.jsonl) | [MP4](videos/v7x_i2v_1344x768_345f_5steps_fp8_turbo-fl2v-v1.2.mp4) |
| 5 (4) | Turbo `fl2v v1.2` | FP8 | 6.0 | 3 | **80.345 s** | 84.886 s | [jsonl](results/v7x_1host/sweeps_fp8.jsonl) | [MP4](videos/v7x_i2v_1344x768_345f_5steps_fp8_turbo-fl2v-v1.2_shift6.mp4) |
| 5 (4) | Turbo `fl2v v1.2` | BF16 | 6.0 | 3 | **82.022 s** | 86.396 s | [jsonl](results/v7x_1host/sweeps_bf16.jsonl) | [MP4](videos/v7x_i2v_1344x768_345f_5steps_bf16_turbo-fl2v-v1.2_shift6.mp4) |
| 26 (25) | FL2VA base | FP8 | 12.0 | 1 | **353.289 s** | 359.347 s | [jsonl](results/v7x_1host/sweeps_fp8.jsonl) | [MP4](videos/v7x_i2v_1344x768_345f_26steps_fp8_base.mp4) |
| 26 (25) | FL2VA base | BF16 | 12.0 | 1 | **382.555 s** | 387.267 s | [jsonl](results/v7x_1host/sweeps_bf16.jsonl) | [MP4](videos/v7x_i2v_1344x768_345f_26steps_bf16_base.mp4) |

### 5.2 Video length (I2V, 1344×768, Turbo `fl2v v1.2`, 5 steps, flow shift 6.0)

| Frames (24 fps) | Output length | Precision | Runs | Median `inference_time_s` | Median end-to-end | Record | Video |
| ---: | ---: | :--- | ---: | ---: | ---: | :--- | :--- |
| 124 | 5.17 s | FP8 | 3 | **31.317 s** | 33.346 s | [jsonl](results/v7x_1host/sweeps_fp8.jsonl) | [MP4](videos/v7x_i2v_1344x768_124f_5steps_fp8_turbo-fl2v-v1.2_shift6.mp4) |
| 124 | 5.17 s | BF16 | 3 | **30.622 s** | 32.831 s | [jsonl](results/v7x_1host/sweeps_bf16.jsonl) | [MP4](videos/v7x_i2v_1344x768_124f_5steps_bf16_turbo-fl2v-v1.2_shift6.mp4) |
| 243 | 10.13 s | FP8 | 2 | **53.521 s** | 56.992 s | [jsonl](results/v7x_1host/sweeps_fp8.jsonl) | — |
| 345 | 14.375 s | FP8 | 3 | **80.345 s** | 84.886 s | [jsonl](results/v7x_1host/sweeps_fp8.jsonl) | [MP4](videos/v7x_i2v_1344x768_345f_5steps_fp8_turbo-fl2v-v1.2_shift6.mp4) |
| 362 | 15.08 s | FP8 | 2 | **91.675 s** | 96.809 s | [jsonl](results/v7x_1host/sweeps_fp8.jsonl) | — |

### 5.3 FastH3 distilled checkpoint (T2VA, 1344×768, FP8, `num_inference_steps = 5`, 4 DiT passes)

| Frames (24 fps) | Output length | Runs | Median `inference_time_s` | Median end-to-end | First request | Record | Video |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| 124 | 5.17 s | 3 | **29.043 s** | 30.756 s | 36.245 s | [jsonl](results/v7x_1host/fasth3_t2va_fp8.jsonl) | [MP4](videos/v7x_fasth3_t2va_1344x768_124f_5steps_fp8.mp4) |
| 345 | 14.375 s | 2 | **76.783 s** | 81.198 s | 432.695 s | [jsonl](results/v7x_1host/fasth3_t2va_fp8.jsonl) | [MP4](videos/v7x_fasth3_t2va_1344x768_345f_5steps_fp8.mp4) |
| 362 | 15.08 s | 2 | **80.966 s** | 85.352 s | 469.01 s | [jsonl](results/v7x_1host/fasth3_t2va_fp8.jsonl) | — |

---

## 6. Multi-host throughput: TPU v7x

Each 4-chip host runs one server replica (Ulysses 8 across its 8 TensorCores). Concurrent client requests are load-balanced across the replicas, and all replicas share the CPU MP4 encoding service.

The workload is I2V at 1344×768: FL2VA + Turbo `fl2v v1.2` (merged), online FP8, `num_inference_steps = 5`.

| Topology | Chips | Replicas | Output | Concurrency | Requests | Wall time | Throughput (videos/min) | Video-seconds per wall-second | p50 end-to-end | p90 end-to-end | Median `inference_time_s` | Record |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 2 hosts | 8 | 2 | 345 frames (14.375 s) | 2 | 4 | 172.71 s | **1.390** | 0.333 | 85.291 s | 87.889 s | 80.265 s | [jsonl](results/v7x_2host/dp2_throughput_fp8.jsonl) |
| 2 hosts | 8 | 2 | 124 frames (5.17 s) | 4 | 8 | 195.49 s | **2.455** | 0.211 | 96.543 s | 98.260 s | 30.373 s | [jsonl](results/v7x_2host/dp2_throughput_fp8.jsonl) |
| 4 hosts | 16 | 4 | 345 frames (14.375 s) | 4 | 8 | 196.81 s | **2.439** | 0.584 | 93.373 s | 100.914 s | 88.843 s | [jsonl](results/v7x_4host/dp4_throughput_fp8.jsonl) |
| 4 hosts | 16 | 4 | 345 frames (14.375 s) | 8 | 12 | 304.11 s | **2.368** | 0.567 | 175.799 s | 202.598 s | 89.799 s | [jsonl](results/v7x_4host/dp4_throughput_fp8.jsonl) |
| 4 hosts | 16 | 4 | 124 frames (5.17 s) | 4 | 12 | 140.91 s | **5.110** | 0.440 | 40.989 s | 49.327 s | 38.741 s | [jsonl](results/v7x_4host/dp4_throughput_fp8.jsonl) |
| 4 hosts | 16 | 4 | 124 frames (5.17 s) | 8 | 16 | 188.20 s | **5.101** | 0.439 | 72.936 s | 99.101 s | 40.247 s | [jsonl](results/v7x_4host/dp4_throughput_fp8.jsonl) |

---

## 7. TPU v5p, one host (4 chips)

Configuration:
- BF16, with Ulysses 4 across the host's 4 chips.
- `--encoder-parallel=fold` and SDPA attention.
- The text encoder and VAE are offloaded between uses.
- The 50-step T2VA server also shards the DiT weights with FSDP.

### 7.1 Workloads from §2 (identical repeated requests)

| Workload | Output | Weights and precision | Warm runs (s) | Median `inference_time_s` | Median end-to-end | Encode + I/O | First request | Peak memory (MB) | Record | Video |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| I2V (§2.2), 15 s, 4 steps | 1344×768, 345 frames | FL2VA base, BF16 | 153.510 / 152.794 / 160.661 | **153.510 s** | 159.095 s | 5.585 s | 1,687.564 s | 73,307 | [jsonl](results/v5p_1host/i2v_4steps_bf16.jsonl) | [MP4](videos/v5p_i2v_1344x768_345f_4steps_bf16_base.mp4) |
| I2V (§2.2), 15 s, 4 steps | 1344×768, 345 frames | FL2VA + Turbo `fl2v v1.2 768p` (merged), BF16 | 143.815 / 147.202 / 146.113 | **146.113 s** | 150.710 s | 4.597 s | 1,635.827 s | 73,307 | [jsonl](results/v5p_1host/i2v_4steps_bf16.jsonl) | [MP4](videos/v5p_i2v_1344x768_345f_4steps_bf16_turbo-fl2v-v1.2.mp4) |
| T2VA (§2.3), 5 s, 50 steps | 1344×768, 124 frames | FL2VA base, BF16 (lossless) | 327.131 / 330.833 / 333.977 | **330.833 s** | 332.936 s | 2.103 s | 948.619 s | 32,430 | [jsonl](results/v5p_1host/t2va_50steps_bf16.jsonl) | [MP4](videos/v5p_t2va_1344x768_124f_50steps_bf16_base.mp4) |

**Notes:**
- For the T2VA server, the first request was a 5-step request with the same output shape.
- Ref2VA with 6 references (§2.1) was not completed on a single v5p host in this round, so it is not included.

### 7.2 Additional TPU v5p points

| Workload | Output | `num_inference_steps` (DiT passes) | Weights | Flow shift | Runs | Median `inference_time_s` | Median end-to-end | First request | Record | Video |
| :--- | :--- | :---: | :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| I2V | 1344×768, 345 frames | 5 (4) | FL2VA + Turbo `fl2v v1.2` (merged) | 6.0 | 3 | **175.179 s** | 180.335 s | 1,643.935 s | [jsonl](results/v5p_1host/i2v_5steps_bf16.jsonl) | [MP4](videos/v5p_i2v_1344x768_345f_5steps_bf16_turbo-fl2v-v1.2_shift6.mp4) |
| T2VA (FastH3) | 1344×768, 124 frames | 5 (4) | `FastVideo/FastH3` | default | 3 | **105.089 s** | 107.130 s | 1,015.31 s | [jsonl](results/v5p_1host/fasth3_t2va_bf16.jsonl) | [MP4](videos/v5p_fasth3_t2va_1344x768_124f_5steps_bf16.mp4) |

---

## 8. Output verification (`ffprobe`)

| Video | Size (bytes) | Duration | Video stream | Audio stream |
| :--- | ---: | ---: | :--- | :--- |
| [v5p_fasth3_t2va_1344x768_124f_5steps_bf16.mp4](videos/v5p_fasth3_t2va_1344x768_124f_5steps_bf16.mp4) | 1,414,186 | 5.175 s | `h264` 1344×768, 24 fps, 124 frames | `aac` 32 kHz, 2 ch |
| [v5p_i2v_1344x768_345f_4steps_bf16_base.mp4](videos/v5p_i2v_1344x768_345f_4steps_bf16_base.mp4) | 11,809,982 | 14.375 s | `h264` 1344×768, 24 fps, 345 frames | `aac` 32 kHz, 2 ch |
| [v5p_i2v_1344x768_345f_4steps_bf16_turbo-fl2v-v1.2.mp4](videos/v5p_i2v_1344x768_345f_4steps_bf16_turbo-fl2v-v1.2.mp4) | 8,870,951 | 14.375 s | `h264` 1344×768, 24 fps, 345 frames | `aac` 32 kHz, 2 ch |
| [v5p_i2v_1344x768_345f_5steps_bf16_turbo-fl2v-v1.2_shift6.mp4](videos/v5p_i2v_1344x768_345f_5steps_bf16_turbo-fl2v-v1.2_shift6.mp4) | 8,156,872 | 14.375 s | `h264` 1344×768, 24 fps, 345 frames | `aac` 32 kHz, 2 ch |
| [v5p_t2va_1344x768_124f_50steps_bf16_base.mp4](videos/v5p_t2va_1344x768_124f_50steps_bf16_base.mp4) | 1,352,762 | 5.175 s | `h264` 1344×768, 24 fps, 124 frames | `aac` 32 kHz, 2 ch |
| [v7x_fasth3_t2va_1344x768_124f_5steps_fp8.mp4](videos/v7x_fasth3_t2va_1344x768_124f_5steps_fp8.mp4) | 1,526,973 | 5.175 s | `h264` 1344×768, 24 fps, 124 frames | `aac` 32 kHz, 2 ch |
| [v7x_fasth3_t2va_1344x768_345f_5steps_fp8.mp4](videos/v7x_fasth3_t2va_1344x768_345f_5steps_fp8.mp4) | 6,685,022 | 14.375 s | `h264` 1344×768, 24 fps, 345 frames | `aac` 32 kHz, 2 ch |
| [v7x_i2v_1344x768_124f_5steps_bf16_turbo-fl2v-v1.2_shift6.mp4](videos/v7x_i2v_1344x768_124f_5steps_bf16_turbo-fl2v-v1.2_shift6.mp4) | 3,453,796 | 5.175 s | `h264` 1344×768, 24 fps, 124 frames | `aac` 32 kHz, 2 ch |
| [v7x_i2v_1344x768_124f_5steps_fp8_turbo-fl2v-v1.2_shift6.mp4](videos/v7x_i2v_1344x768_124f_5steps_fp8_turbo-fl2v-v1.2_shift6.mp4) | 3,799,863 | 5.175 s | `h264` 1344×768, 24 fps, 124 frames | `aac` 32 kHz, 2 ch |
| [v7x_i2v_1344x768_345f_26steps_bf16_base.mp4](videos/v7x_i2v_1344x768_345f_26steps_bf16_base.mp4) | 9,352,527 | 14.375 s | `h264` 1344×768, 24 fps, 345 frames | `aac` 32 kHz, 2 ch |
| [v7x_i2v_1344x768_345f_26steps_fp8_base.mp4](videos/v7x_i2v_1344x768_345f_26steps_fp8_base.mp4) | 9,881,381 | 14.375 s | `h264` 1344×768, 24 fps, 345 frames | `aac` 32 kHz, 2 ch |
| [v7x_i2v_1344x768_345f_4steps_bf16_base.mp4](videos/v7x_i2v_1344x768_345f_4steps_bf16_base.mp4) | 12,686,447 | 14.375 s | `h264` 1344×768, 24 fps, 345 frames | `aac` 32 kHz, 2 ch |
| [v7x_i2v_1344x768_345f_4steps_fp8_base.mp4](videos/v7x_i2v_1344x768_345f_4steps_fp8_base.mp4) | 13,275,598 | 14.375 s | `h264` 1344×768, 24 fps, 345 frames | `aac` 32 kHz, 2 ch |
| [v7x_i2v_1344x768_345f_4steps_fp8_turbo-fl2v-v1.2.mp4](videos/v7x_i2v_1344x768_345f_4steps_fp8_turbo-fl2v-v1.2.mp4) | 10,762,143 | 14.375 s | `h264` 1344×768, 24 fps, 345 frames | `aac` 32 kHz, 2 ch |
| [v7x_i2v_1344x768_345f_4steps_fp8_turbo-fl2v-v1.2_shift6.mp4](videos/v7x_i2v_1344x768_345f_4steps_fp8_turbo-fl2v-v1.2_shift6.mp4) | 11,340,321 | 14.375 s | `h264` 1344×768, 24 fps, 345 frames | `aac` 32 kHz, 2 ch |
| [v7x_i2v_1344x768_345f_5steps_bf16_turbo-fl2v-v1.2_shift6.mp4](videos/v7x_i2v_1344x768_345f_5steps_bf16_turbo-fl2v-v1.2_shift6.mp4) | 9,266,202 | 14.375 s | `h264` 1344×768, 24 fps, 345 frames | `aac` 32 kHz, 2 ch |
| [v7x_i2v_1344x768_345f_5steps_fp8_turbo-fl2v-v1.2.mp4](videos/v7x_i2v_1344x768_345f_5steps_fp8_turbo-fl2v-v1.2.mp4) | 10,315,259 | 14.375 s | `h264` 1344×768, 24 fps, 345 frames | `aac` 32 kHz, 2 ch |
| [v7x_i2v_1344x768_345f_5steps_fp8_turbo-fl2v-v1.2_shift6.mp4](videos/v7x_i2v_1344x768_345f_5steps_fp8_turbo-fl2v-v1.2_shift6.mp4) | 10,225,390 | 14.375 s | `h264` 1344×768, 24 fps, 345 frames | `aac` 32 kHz, 2 ch |
| [v7x_ref2va_1ref_1344x768_124f_50steps_bf16_base.mp4](videos/v7x_ref2va_1ref_1344x768_124f_50steps_bf16_base.mp4) | 2,444,256 | 5.175 s | `h264` 1344×768, 24 fps, 124 frames | `aac` 32 kHz, 2 ch |
| [v7x_ref2va_1ref_1344x768_124f_50steps_fp8_base.mp4](videos/v7x_ref2va_1ref_1344x768_124f_50steps_fp8_base.mp4) | 2,367,762 | 5.175 s | `h264` 1344×768, 24 fps, 124 frames | `aac` 32 kHz, 2 ch |
| [v7x_ref2va_6refs_1792x768_345f_4steps_fp8_turbo-ref2v-v0.1.mp4](videos/v7x_ref2va_6refs_1792x768_345f_4steps_fp8_turbo-ref2v-v0.1.mp4) | 14,670,940 | 14.375 s | `h264` 1792×768, 24 fps, 345 frames | `aac` 32 kHz, 2 ch |
| [v7x_t2va_1344x768_124f_50steps_bf16_base.mp4](videos/v7x_t2va_1344x768_124f_50steps_bf16_base.mp4) | 1,427,139 | 5.175 s | `h264` 1344×768, 24 fps, 124 frames | `aac` 32 kHz, 2 ch |
| [v7x_t2va_1344x768_124f_50steps_fp8_base.mp4](videos/v7x_t2va_1344x768_124f_50steps_fp8_base.mp4) | 1,498,760 | 5.175 s | `h264` 1344×768, 24 fps, 124 frames | `aac` 32 kHz, 2 ch |
