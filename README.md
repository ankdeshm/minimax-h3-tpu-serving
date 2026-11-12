# MiniMax-H3 Serving on Google Cloud TPU v7x and v5p

Single-host latency and multi-host serving throughput benchmarks for [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) (joint video + stereo audio generation) on Google Cloud TPU `v7x` and `v5p` using `sglang-torchtpu`.

**Summary:** On a single 4-chip TPU `v7x` host (`8` TensorCores, `768 GiB` HBM) with Ulysses-8 sequence parallelism, Pallas Splash Attention, folded Qwen3-VL encoder sharding, and online FP8 DiT quantization, **MiniMax-H3 generates 15 s 768p image-to-video + audio (`345` frames, 4 steps) in 67.68 s (`2.16×` faster than 146.11 s in `bf16` on a 4-chip TPU `v5p` host) and 50-step 5 s 768p text-to-video + audio (`124` frames) in 122.03 s (`1.08×` faster than 132.24 s in `v7x` `bf16` and `2.71×` faster than 330.83 s on a 4-chip `v5p` host).** Scaling data-parallel replicas across a 4-host `v7x-16` slice reaches **5.11 videos/min** on 5 s clips (`2.08×` higher than 2.46 videos/min on 2 hosts) and **2.44 videos/min** on 15 s clips (`1.75×` higher than 1.39 videos/min on 2 hosts).

---

## 1. Model, Framework, and Hardware

- **Model:** [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3) (`FL2VA` image/text-to-video+audio and `Ref2VA` reference-to-video+audio pipelines), optional 4-step [`lightx2v/Minimax-h3-Turbo`](https://huggingface.co/lightx2v/Minimax-h3-Turbo) LoRA adapters, and the distilled 5-step [`FastVideo/FastH3`](https://huggingface.co/FastVideo/FastH3) checkpoint.
  - **Parameters & Weights:** **33.14B-parameter** omni-modal DiT (`19.27B` active per-token trunk + `13.05B` modality AdaLN) paired with a `66.7 GB` Qwen3-VL multimodal encoder, 3D video VAE, and stereo audio VAE. Evaluated in lossless `bfloat16`/`float32` and online `fp8` (`--quantization=fp8` on DiT linear layers, with Turbo LoRA merged into `bf16` prior to `fp8` quantization and attention/norms kept in `bf16`).
  - **Architecture:** 50-block single-stream DiT (`hidden_size = 5,376`, `56` heads × `128` head dim, `14,336` SwiGLU FFN) running full bidirectional dense self-attention (`cache_dit` disabled) over packed text, audio, and 3D video latents (`N` scheduler steps = `N - 1` DiT passes), producing 24 fps H.264 video with synchronized 32 kHz stereo AAC audio.
- **Framework:** `sglang-torchtpu` (`TorchTPU` 0.2.0, `libtpu` 0.0.50, `JAX` 0.11.2), Pallas Splash Attention (`--attention-backend=tpu_splash_attn`), Ulysses sequence parallelism (`--ulysses-degree=8` on `v7x`, `7` heads/core; `--ulysses-degree=4` on `v5p`), folded encoder tensor parallelism (`--encoder-parallel=fold`), and a disaggregated CPU FFmpeg MP4 muxing service ([`scripts/video_encode_server.py`](./scripts/video_encode_server.py)).
- **Hardware:** Google Cloud TPU on GKE:
  - **TPU v7x (Ironwood):** `tpu7x-standard-4t` (`4` chips = `8` TensorCores and `768 GiB` HBM3e per host, `96 GiB`/TensorCore), tested at 1 host (`4` chips), 2 hosts (`8` chips), and 4 hosts (`16` chips).
  - **TPU v5p:** `ct5p-hightpu-4t` (`4` chips = `380 GiB` HBM2e per host, `95 GiB`/chip), tested at 1 host (`4` chips).

---

## 2. Main Results

### A. Single-Host Latency (`1` Host = `4` Chips, Warm Server, Concurrency 1)

`inference_time_s` is the server-reported pipeline latency through decoded video frames and audio (excluding HTTP transport and CPU MP4 muxing); end-to-end client time includes MP4 muxing. Every row links to its generated `.mp4` in [`videos/`](./videos/).

| Workload & Output (`24 fps` + `32 kHz` Stereo Audio) | Steps (`DiT Passes`) | Platform & Precision | Median `inference_time_s` | Median End-to-End | Output Video | Comparison |
| :--- | :---: | :--- | ---: | ---: | :--- | :--- |
| **I2V (`1344×768`, `345` frames / `14.4 s`)** — FL2VA + Turbo `fl2v v1.2` | 4 (`3`) | **TPU v7x (`fp8`)** | **67.68 s** | **72.35 s** | [MP4](./videos/v7x_i2v_1344x768_345f_4steps_fp8_turbo-fl2v-v1.2.mp4) | **2.16× faster** than 4-chip `v5p` `bf16` (`146.11 s`) |
| I2V (`1344×768`, `345` frames / `14.4 s`) — FL2VA base | 4 (`3`) | TPU v7x (`fp8` / `bf16`) | 68.10 s / 67.99 s | 72.89 s / 72.88 s | [FP8](./videos/v7x_i2v_1344x768_345f_4steps_fp8_base.mp4) · [BF16](./videos/v7x_i2v_1344x768_345f_4steps_bf16_base.mp4) | **2.25× faster** than 4-chip `v5p` `bf16` (`153.51 s`) |
| I2V (`1344×768`, `345` frames / `14.4 s`) — FL2VA base / Turbo | 4 (`3`) | TPU v5p (`bf16`) | 153.51 s / 146.11 s | 159.10 s / 150.71 s | [Base](./videos/v5p_i2v_1344x768_345f_4steps_bf16_base.mp4) · [Turbo](./videos/v5p_i2v_1344x768_345f_4steps_bf16_turbo-fl2v-v1.2.mp4) | 4-chip `v5p` `bf16` baseline (`1.00×`) |
| **T2VA (`1344×768`, `124` frames / `5.2 s`)** — FL2VA base | 50 (`49`) | **TPU v7x (`fp8`)** | **122.03 s** | **123.84 s** | [MP4](./videos/v7x_t2va_1344x768_124f_50steps_fp8_base.mp4) | **1.08× faster** than `v7x` `bf16`; **2.71× faster** than `v5p` `bf16` |
| T2VA (`1344×768`, `124` frames / `5.2 s`) — FL2VA base | 50 (`49`) | TPU v7x (`bf16`) | 132.24 s | 134.68 s | [MP4](./videos/v7x_t2va_1344x768_124f_50steps_bf16_base.mp4) | **2.50× faster** than 4-chip `v5p` `bf16` (`330.83 s`) |
| T2VA (`1344×768`, `124` frames / `5.2 s`) — FL2VA base | 50 (`49`) | TPU v5p (`bf16`) | 330.83 s | 332.94 s | [MP4](./videos/v5p_t2va_1344x768_124f_50steps_bf16_base.mp4) | 4-chip `v5p` `bf16` 50-step baseline (`1.00×`) |
| **T2VA (`1344×768`, `124` frames / `5.2 s`)** — `FastH3` distilled | 5 (`4`) | **TPU v7x (`fp8`)** | **29.04 s** | **30.76 s** | [MP4](./videos/v7x_fasth3_t2va_1344x768_124f_5steps_fp8.mp4) | **3.62× faster** than `v5p` `FastH3` `bf16` (`105.09 s`) |
| **Ref2VA (`1344×768`, `124` frames / `5.2 s`, 1 image ref)** | 50 (`49`) | **TPU v7x (`fp8` / `bf16`)** | **206.34 s** / 215.56 s | **208.53 s** / 217.96 s | [FP8](./videos/v7x_ref2va_1ref_1344x768_124f_50steps_fp8_base.mp4) · [BF16](./videos/v7x_ref2va_1ref_1344x768_124f_50steps_bf16_base.mp4) | `fp8` is **4.5% faster** (`1.045×`) than `v7x` `bf16` |
| **Ref2VA (`1792×768` `21:9`, `345` frames, 1 video + 5 image refs)** | 4 (`3`) | **TPU v7x (`fp8`, Turbo `v0.1`)** | **277.64 s** | **283.89 s** | [MP4](./videos/v7x_ref2va_6refs_1792x768_345f_4steps_fp8_turbo-ref2v-v0.1.mp4) | Full 6-reference joint video + audio context (`70.2 GB`/core) |

*Notes:* Warm medians after initial XLA compilation (cold first-request compile times, per-stage timings, and 26-step / multi-duration sweeps are in [`APPENDIX.md`](./APPENDIX.md)). At 4 steps (`3` DiT passes), VAE decoding (`~19.7 s`) and encoder/host orchestration (`~10.7 s`) account for `45%` of total time, so `fp8` and `bf16` match closely (`68.10 s` vs. `67.99 s`), whereas at 26–50 steps `fp8` speeds up total inference by `4.5%–8.4%` and reduces peak TensorCore HBM by `7–17 GB`.

### B. Multi-Host Serving Throughput on TPU `v7x` (I2V `1344×768`, 5 Steps, Turbo `fl2v v1.2`, Online `fp8`)

Each 4-chip `v7x` host runs an independent Ulysses-8 `sglang-torchtpu` replica sharing the CPU MP4 encoding service ([`results/v7x_2host/`](./results/v7x_2host/), [`results/v7x_4host/`](./results/v7x_4host/)):

| Slice Topology | Hosts (`Chips` / `Replicas`) | Output Length | Concurrency | Throughput | `p50` End-to-End | Comparison |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- |
| `v7x-8` | 2 hosts (`8` chips / `2` replicas) | `345` frames (`14.375 s`) | 2 | 1.390 videos/min | 85.29 s | 2-host baseline (`1.00×`) |
| **`v7x-16`** | **4 hosts (`16` chips / `4` replicas)** | **`345` frames (`14.375 s`)** | **4** | **2.439 videos/min** | **93.37 s** | **1.75×** higher throughput vs. 2 hosts (`88%` scaling efficiency) |
| `v7x-8` | 2 hosts (`8` chips / `2` replicas) | `124` frames (`5.17 s`) | 4 | 2.455 videos/min | 96.54 s | 2-host baseline (`1.00×`) |
| **`v7x-16`** | **4 hosts (`16` chips / `4` replicas)** | **`124` frames (`5.17 s`)** | **4** | **5.110 videos/min** | **40.99 s** | **2.08×** higher throughput and **2.36× lower `p50`** vs. 2 hosts (`c=4`) |

---

## 3. Reproducing the Main Results

Stage `MiniMaxAI/MiniMax-H3` and `lightx2v/Minimax-h3-Turbo` to your GCS bucket, update `<YOUR_GCS_BUCKET>` and `<YOUR_ARTIFACT_REGISTRY>` in [`k8s/`](./k8s/), and deploy the CPU MP4 muxing service ([`scripts/video_encode_server.py`](./scripts/video_encode_server.py)) plus the `sglang-torchtpu` server or benchmark jobs ([`scripts/run_minimax_h3_tpu_benchmark.py`](./scripts/run_minimax_h3_tpu_benchmark.py)):

```bash
# 1. Stage checkpoints to Cloud Storage
hf download MiniMaxAI/MiniMax-H3 --local-dir ./MiniMax-H3
hf download lightx2v/Minimax-h3-Turbo --local-dir ./Minimax-h3-Turbo
gcloud storage cp -r ./MiniMax-H3 gs://${GCS_BUCKET}/models/MiniMaxAI/MiniMax-H3
gcloud storage cp -r ./Minimax-h3-Turbo gs://${GCS_BUCKET}/models/lightx2v/Minimax-h3-Turbo

# 2. Deploy disaggregated CPU MP4 encoder + single-host TPU v7x SGLang server
kubectl apply -f k8s/video-encode-svc.yaml
kubectl apply -f k8s/sglang-minimax-h3-server.yaml

# 3. Run single-host latency, stage-profiling, and multi-host throughput jobs
kubectl apply -f k8s/minimax-h3-v7x-single-host.yaml
kubectl apply -f k8s/minimax-h3-v7x-8.yaml
kubectl apply -f k8s/minimax-h3-v7x-16.yaml
kubectl apply -f k8s/minimax-h3-v5p-16.yaml
```

---

## 4. Detailed Ablations, Engineering Notes, and Raw Logs

- **[`APPENDIX.md`](./APPENDIX.md):** Complete workload prompts and SHA-256 input checksums, cold compile latencies, device-synchronized per-stage breakdowns, step-count (`4 / 5 / 26`) and duration (`124–362` frames) sweeps, `FastH3` and `v5p` results, and `ffprobe` verification for all 23 generated MP4s.
- **[`results/`](./results/) & [`videos/`](./videos/):** Raw JSONL records (`v7x_1host/`, `v7x_2host/`, `v7x_4host/`, `v5p_1host/`), stage profile JSONs (`results/perf/`), and generated H.264 + stereo AAC videos.
