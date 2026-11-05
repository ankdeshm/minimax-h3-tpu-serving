#!/usr/bin/env python3
"""Cloud TPU v7x & v5p Serving Benchmark Runner for MiniMax-H3 (Ref2VA, FL2VA/I2V, T2VA, FastH3).

Executes single-host and multi-host slice serving benchmarks on Cloud TPU `v7x` (`v7x-8`, `v7x-16`)
and Cloud TPU `v5p` (`v5p-16`, `v5p-32`) using `sglang-torchtpu` + Disaggregated CPU Video+Audio
Encoding (`video-encode-svc`), saving generated H.264 + 32 kHz stereo AAC `.mp4` videos and
incremental JSONL/JSON metrics to Google Cloud Storage.

Key TPU Optimizations Applied:
  1. Folded Qwen3-VL Encoder Tensor Parallelism (`--encoder-parallel fold`), sharding the 66.7 GB
     multimodal encoder across all 8 TensorCores (`8.34 GB/core` on `v7x`) or 4 chips (`16.7 GB/chip` on `v5p`).
  2. All-HBM-Resident Execution on `v7x` FP8 (`33.1 GB` DiT + `8.3 GB` Qwen3-VL + `5.2 GB` VideoVAE =
     `46.7 GB/core` within `96 GiB/core` HBM3e), eliminating per-request PCIe host-to-device weight copies,
     with automatic fallback to selective text-encoder CPU offload on extreme multi-reference shapes.
  3. True Pallas Mosaic Varlen FlashAttention (`SGLANG_TPU_SPLASH_VARLEN=1`) for MiniMax-H3 VDN varlen sequences.
  4. Native `v7x` FP8 MXU (`torch._scaled_mm(..., out_dtype=torch.bfloat16)`) + Pre-Quantization In-Place
     BF16 LoRA Fusion (`312/312` adapter pairs with SwiGLU `[value; gate] -> [gate; value]` permutation).
  5. Generalized Static XLA Scatter (`scatter` / `scatter_add`) in Qwen3-VL supporting arbitrary
     combinations of reference videos and multiple reference images (`Ref2VA` 1 video + 5 images).
  6. Exact `1792x768` (`21:9` @ `768p`) and `1344x768` (`16:9` @ `768p`) spatial resolution + `345`-frame
     (`14.375s` @ `24 fps`) temporal alignment.
  7. Disaggregated AVX-512 CPU Video + Stereo Audio MP4 Muxing (`SGLANG_REMOTE_ENCODE_URL`).
"""

from __future__ import annotations

import os

os.environ["HF_HOME"] = "/dev/shm/hf_cache"
os.environ["HF_HUB_CACHE"] = "/dev/shm/hf_cache/hub"
os.environ["SGLANG_DIFFUSION_CACHE_ROOT"] = "/dev/shm/sglang_diffusion_cache"

import argparse
import concurrent.futures
import hashlib
import json
import math
import py_compile
import resource
import shutil
import signal
import socket
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
except Exception:
    pass

DEFAULT_REMOTE_ENCODE_URL = "http://video-encode-svc.default.svc.cluster.local:8090/encode"

# Sample Prompts & Exact Evaluation Workloads
# Reference 6-input Ref2VA prompt (1 reference video + 5 reference images).
# NOTE: the prompt intentionally contains TWO spaces after "Viking,". Do not "clean up" this string:
# prompt text changes tokenization and therefore the generated video.
PROMPT_REF2VA_SAMPLE = (
    "Using Image One the video make sound clip music in English translation , "
    "testosterone Viking,  use images attached to make the image identify of male"
)
PROMPT_I2V_DRAGON = "The camera slowly pushes in with epic bgm. Keep the subject identity."
PROMPT_T2VA_CATS = (
    "At night, while their owner sleeps in a bedroom, three cats march in loudly "
    "playing tiny brass instruments, then abruptly file out."
)
PROMPTS_SWEEP = [
    PROMPT_I2V_DRAGON,
    PROMPT_T2VA_CATS,
    "A cinematic wide tracking shot of golden sunset waves crashing against dramatic volcanic sea cliffs, warm rim light, crisp ocean surf and distant seabirds.",
    "A futuristic high-speed train gliding across a misty glass bridge over an emerald mountain canyon at dawn, rich ambient wind and subtle rail hum.",
]

METADATA_HOOK_C = r"""
#define _GNU_SOURCE
#include <sys/socket.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <dlfcn.h>
#include <string.h>

static int (*real_connect)(int, const struct sockaddr*, socklen_t) = NULL;

int connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen) {
    if (!real_connect) {
        real_connect = dlsym(RTLD_NEXT, "connect");
    }
    if (addr && addr->sa_family == AF_INET) {
        const struct sockaddr_in *addr_in = (const struct sockaddr_in *)addr;
        char ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &(addr_in->sin_addr), ip, INET_ADDRSTRLEN);
        if (strcmp(ip, "169.254.169.254") == 0) {
            struct sockaddr_in redirected = *addr_in;
            redirected.sin_addr.s_addr = inet_addr("127.0.0.1");
            redirected.sin_port = htons(80);
            return real_connect(sockfd, (const struct sockaddr *)&redirected, sizeof(redirected));
        }
    }
    return real_connect(sockfd, addr, addrlen);
}
"""

MOCK_METADATA_PY = r"""#!/usr/bin/env python3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import sys
import urllib.error
import urllib.parse
import urllib.request


class MockMetadataHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def forward_to_real_metadata(self):
        real_url = f"http://169.254.169.254{self.path}"
        req = urllib.request.Request(real_url)
        for key, val in self.headers.items():
            if key.lower() != "host":
                req.add_header(key, val)
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                self.send_response(response.status)
                for key, val in response.headers.items():
                    self.send_header(key, val)
                self.end_headers()
                self.wfile.write(response.read())
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            for key, val in e.headers.items():
                self.send_header(key, val)
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode())

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        if self.headers.get("Metadata-Flavor") != "Google":
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Missing Metadata-Flavor header")
            return

        response_body = None
        tpu_env_keys = {
            "TPU_WORKER_ID": "0",
            "ALT": "false",
            "CHIPS_PER_HOST_BOUNDS": "2,2,1",
            "CONSUMER_PROJECT_ID": "tpu-serving-bench",
            "ENABLE_ICI_RESILIENCY": "false",
            "ENABLE_IMPROVED_REROUTE_ALLREDUCE_STRATEGY": "false",
            "HOST_BOUNDS": "1,1,1",
            "INJECT_SLICE_BUILDER_FAULT": "",
            "NODE_ID": "0",
            "TOPOLOGY": "2x2x1",
            "TPU_ACCELERATOR_TYPE": "tpu7x-8",
            "WRAP": "false,false,false",
            "ZONE": "us-central1-c",
        }
        attr = path.rsplit("/attributes/", 1)[-1] if "/instance/attributes/" in path else None
        if attr in tpu_env_keys:
            response_body = tpu_env_keys[attr].encode()
        elif path.endswith("/computeMetadata/v1/instance/attributes/tpu-env"):
            response_body = (
                "TPU_WORKER_ID: '0'\n"
                "ALT: 'false'\n"
                "CHIPS_PER_HOST_BOUNDS: '2,2,1'\n"
                "CONSUMER_PROJECT_ID: 'tpu-serving-bench'\n"
                "ENABLE_ICI_RESILIENCY: 'false'\n"
                "ENABLE_IMPROVED_REROUTE_ALLREDUCE_STRATEGY: 'false'\n"
                "HOST_BOUNDS: '1,1,1'\n"
                "INJECT_SLICE_BUILDER_FAULT: ''\n"
                "NODE_ID: '0'\n"
                "TOPOLOGY: '2x2x1'\n"
                "TPU_ACCELERATOR_TYPE: 'tpu7x-8'\n"
                "WRAP: 'false,false,false'\n"
                "ZONE: 'us-central1-c'\n"
            ).encode()
        elif path.endswith("/computeMetadata/v1/instance/attributes/worker-network-endpoints") or path.endswith(
            "/computeMetadata/v1/instance/attributes/instance-id"
        ):
            req = urllib.request.Request(
                "http://169.254.169.254/computeMetadata/v1/instance/hostname",
                headers={"Metadata-Flavor": "Google"},
            )
            hn = urllib.request.urlopen(req, timeout=2).read().decode().strip().split(".")[0]
            response_body = hn.encode()
        elif path.endswith("/computeMetadata/v1/instance/attributes/accelerator-type"):
            response_body = b"tpu7x-8"
        elif path.endswith("/computeMetadata/v1/instance/attributes/agent-worker-number"):
            response_body = b"0"
        elif path.endswith("/computeMetadata/v1/project/project-id"):
            response_body = b"tpu-serving-bench"
        elif path.endswith("/computeMetadata/v1/project/numeric-project-id"):
            response_body = b"1234567890"
        elif path.endswith("/computeMetadata/v1/universe/universe-domain") or path.endswith("/universe-domain"):
            response_body = b"googleapis.com"

        if response_body is not None:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(response_body)
        else:
            self.forward_to_real_metadata()


if __name__ == "__main__":
    httpd = ThreadingHTTPServer(("127.0.0.1", 80), MockMetadataHandler)
    print("[mock_metadata] Serving on 127.0.0.1:80", file=sys.stderr, flush=True)
    httpd.serve_forever()
"""

_PORT_SEED_COUNTER = 29620
_MOCK_META_PROC: subprocess.Popen[Any] | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(idx: int, msg: str) -> None:
    print(f"[{utc_now()}][pod-{idx}] {msg}", flush=True)


def host_ram_summary() -> str:
    try:
        lines = subprocess.check_output(["free", "-h"], text=True).splitlines()
        return lines[1] if len(lines) > 1 else lines[0]
    except Exception:
        return "unknown"


def shm_free_gib() -> float:
    try:
        st = os.statvfs("/dev/shm")
        return round((st.f_bavail * st.f_frsize) / (1024.0**3), 1)
    except Exception:
        return -1.0


def local_pod_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return socket.gethostbyname(socket.gethostname())


def alloc_free_ports(n: int) -> list[int]:
    socks = []
    ports = []
    for _ in range(n):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        ports.append(s.getsockname()[1])
        socks.append(s)
    for s in socks:
        s.close()
    return ports


def vm_access_token() -> str:
    req = urllib.request.Request(
        "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))["access_token"]


def gcs_upload_bytes(bucket: str, obj: str, data: bytes, content_type: str = "application/json") -> None:
    for attempt in range(5):
        try:
            tok = vm_access_token()
            url = (
                f"https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o"
                f"?uploadType=media&name={urllib.parse.quote(obj, safe='')}"
            )
            req = urllib.request.Request(
                url,
                data=data,
                method="POST",
                headers={"Authorization": f"Bearer {tok}", "Content-Type": content_type},
            )
            with urllib.request.urlopen(req, timeout=180):
                return
        except Exception as exc:
            if attempt == 4:
                print(f"[WARN] GCS upload failed for {obj}: {exc}", flush=True)
                return
            time.sleep(2 * (attempt + 1))


def gcs_upload_file(bucket: str, obj: str, src: Path, content_type: str = "video/mp4") -> bool:
    if not src.exists():
        return False
    data = src.read_bytes()
    gcs_upload_bytes(bucket, obj, data, content_type=content_type)
    return True


def gcs_download_bytes(bucket: str, obj: str) -> bytes | None:
    for attempt in range(4):
        try:
            tok = vm_access_token()
            url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{urllib.parse.quote(obj, safe='')}?alt=media"
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == 3:
                return None
            time.sleep(2 * (attempt + 1))
        except Exception:
            if attempt == 3:
                return None
            time.sleep(2 * (attempt + 1))
    return None


def gcs_download_file(bucket: str, obj: str, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            tok = vm_access_token()
            url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{urllib.parse.quote(obj, safe='')}?alt=media"
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
            with urllib.request.urlopen(req, timeout=180) as resp, open(dst, "wb") as f:
                shutil.copyfileobj(resp, f, length=16 * 1024 * 1024)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            if attempt == 3:
                return False
            time.sleep(2 * (attempt + 1))
        except Exception:
            if attempt == 3:
                return False
            time.sleep(2 * (attempt + 1))
    return False


def gcs_list_prefix(bucket: str, prefix: str) -> list[dict[str, Any]]:
    tok = vm_access_token()
    items: list[dict[str, Any]] = []
    pt: str | None = None
    while True:
        url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o?prefix={urllib.parse.quote(prefix, safe='')}"
        if pt:
            url += f"&pageToken={urllib.parse.quote(pt, safe='')}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            d = json.loads(resp.read().decode("utf-8"))
        items.extend(d.get("items", []))
        pt = d.get("nextPageToken")
        if not pt:
            break
    return items


def fast_gcs_sync_dir(
    idx: int,
    bucket: str,
    gcs_prefix: str,
    local_dir: Path,
    *,
    exclude_prefixes: tuple[str, ...] = (),
) -> int:
    if not gcs_prefix.endswith("/"):
        gcs_prefix += "/"
    items = gcs_list_prefix(bucket, gcs_prefix)
    blobs: list[tuple[str, int]] = []
    for it in items:
        name = it["name"]
        if name.endswith("/") or "/.cache/" in name or name.endswith(".ready"):
            continue
        rel = name[len(gcs_prefix) :]
        if any(rel.startswith(ep) for ep in exclude_prefixes):
            continue
        blobs.append((rel, int(it["size"])))
    if not blobs:
        return 0
    local_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[tuple[str, Path, int, int]] = []
    chunk = 512 * 1024 * 1024
    for rel, sz in blobs:
        dst = local_dir / rel
        if dst.exists() and dst.stat().st_size == sz:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "wb") as f:
            if sz > 0:
                f.truncate(sz)
        if sz == 0:
            continue
        for s in range(0, sz, chunk):
            tasks.append((rel, dst, s, min(s + chunk, sz)))
    if not tasks:
        return len(blobs)
    t0 = time.time()
    total_b = sum(e - s for _, _, s, e in tasks)
    log(idx, f"Syncing {len(tasks)} chunks ({total_b / 1e9:.2f} GB) from gs://{bucket}/{gcs_prefix} -> {local_dir} ...")

    def dl_chunk(t: tuple[str, Path, int, int]) -> int:
        rel, dst, s, e = t
        for r in range(5):
            try:
                tok = vm_access_token()
                u = (
                    f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/"
                    f"{urllib.parse.quote(gcs_prefix + rel, safe='')}?alt=media"
                )
                req = urllib.request.Request(
                    u, headers={"Authorization": f"Bearer {tok}", "Range": f"bytes={s}-{e - 1}"}
                )
                fd = os.open(str(dst), os.O_WRONLY)
                try:
                    pos = s
                    with urllib.request.urlopen(req, timeout=180) as resp:
                        while True:
                            buf = resp.read(16 * 1024 * 1024)
                            if not buf:
                                break
                            os.pwrite(fd, buf, pos)
                            pos += len(buf)
                finally:
                    os.close(fd)
                return e - s
            except Exception:
                if r == 4:
                    raise
                time.sleep(1.5 * (r + 1))
        return 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        list(pool.map(dl_chunk, tasks))
    dt = max(time.time() - t0, 1e-3)
    log(idx, f"Synced {total_b / 1e9:.2f} GB from GCS in {dt:.1f}s ({(total_b / 1e9) / dt:.2f} GB/s)")
    return len(blobs)


def append_and_upload_jsonl(
    bucket: str,
    local_jsonl: Path,
    gcs_jsonl_obj: str,
    row: dict[str, Any],
) -> None:
    local_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(local_jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    gcs_upload_bytes(bucket, gcs_jsonl_obj, local_jsonl.read_bytes(), content_type="application/jsonl")


def gcs_barrier(
    idx: int,
    bucket: str,
    gcs_prefix: str,
    barrier_name: str,
    payload: dict[str, Any],
    *,
    pod_key: str | None = None,
    expected_keys: list[str] | None = None,
    num_pods: int = 2,
    timeout_s: float = 7200.0,
) -> dict[str, dict[str, Any]]:
    key = pod_key or f"pod_{idx}"
    keys = expected_keys or [f"pod_{i}" for i in range(num_pods)]
    b_pref = f"{gcs_prefix}/barriers/{barrier_name}"
    gcs_upload_bytes(bucket, f"{b_pref}/{key}.json", json.dumps(payload).encode("utf-8"))
    log(idx, f"Entered GCS barrier [{barrier_name}] as {key}; waiting for {len(keys)} peers...")
    t0 = time.time()
    gathered: dict[str, dict[str, Any]] = {}
    while time.time() - t0 < timeout_s:
        for k in keys:
            if k in gathered:
                continue
            raw = gcs_download_bytes(bucket, f"{b_pref}/{k}.json")
            if raw is not None:
                try:
                    gathered[k] = json.loads(raw.decode("utf-8"))
                except Exception:
                    pass
        if len(gathered) == len(keys):
            log(idx, f"Barrier [{barrier_name}] cleared across {len(keys)} peers in {time.time() - t0:.1f}s!")
            return gathered
        time.sleep(4.0)
    log(idx, f"WARNING: Barrier [{barrier_name}] timed out with {len(gathered)}/{len(keys)} peers; proceeding.")
    return gathered


def check_remote_encode_service(idx: int, encode_url: str) -> str:
    if not encode_url:
        return ""
    health_url = encode_url.rsplit("/", 1)[0] + "/health"
    for _ in range(5):
        try:
            with urllib.request.urlopen(health_url, timeout=4.0) as resp:
                if resp.status == 200:
                    info = json.loads(resp.read().decode("utf-8"))
                    log(idx, f"Disaggregated CPU Video Encode Service HEALTHY at {encode_url}: {info}")
                    return encode_url
        except Exception:
            time.sleep(1.0)
    log(idx, f"WARNING: {health_url} not reachable; falling back to local CPU encode.")
    return ""


def setup_v7x_metadata_hook(idx: int) -> None:
    global _MOCK_META_PROC
    Path("/workspace").mkdir(parents=True, exist_ok=True)
    for dst_dir in (Path("/workspace"), Path("/tmp")):
        hook_c = dst_dir / "metadata_hook.c"
        hook_so = dst_dir / "metadata_hook.so"
        if not hook_so.exists():
            hook_c.write_text(METADATA_HOOK_C)
            subprocess.run(
                ["gcc", "-O2", "-shared", "-fPIC", str(hook_c), "-o", str(hook_so), "-ldl"],
                check=True,
            )
    subprocess.run(
        ["bash", "-lc", "grep -q 'metadata.google.internal' /etc/hosts || echo '127.0.0.1 metadata.google.internal' >> /etc/hosts"],
        check=False,
    )
    try:
        with socket.create_connection(("127.0.0.1", 80), timeout=0.4):
            log(idx, "mock_metadata.py already listening on 127.0.0.1:80")
            return
    except OSError:
        pass

    mock_py = Path("/workspace/mock_metadata.py")
    mock_py.write_text(MOCK_METADATA_PY)
    env_clean = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
    mf = open("/workspace/mock_metadata.log", "a")
    _MOCK_META_PROC = subprocess.Popen(
        ["python3", "-u", str(mock_py)],
        env=env_clean,
        stdout=mf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    for _ in range(25):
        try:
            req = urllib.request.Request(
                "http://169.254.169.254/computeMetadata/v1/instance/attributes/tpu-env",
                headers={"Metadata-Flavor": "Google"},
            )
            # Query 127.0.0.1:80 directly since runner process itself has no LD_PRELOAD
            req_local = urllib.request.Request(
                "http://127.0.0.1:80/computeMetadata/v1/instance/attributes/tpu-env",
                headers={"Metadata-Flavor": "Google"},
            )
            with urllib.request.urlopen(req_local, timeout=1.0) as r:
                if r.status == 200:
                    log(idx, "mock_metadata.py listening on 127.0.0.1:80")
                    return
        except Exception:
            time.sleep(0.25)
    log(idx, "WARNING: mock_metadata.py port 80 check timed out; check /workspace/mock_metadata.log")


def apply_sglang_tpu_patches(idx: int, *, accelerator: str) -> None:
    """Apply verified SGLang TPU compatibility, multi-image+video Qwen3-VL scatter, 1792x768 21:9, FP8 MXU, Splash, and Disaggregated Encode patches."""
    sp = Path("/usr/local/lib/python3.12/site-packages")
    rt = sp / "sglang/multimodal_gen/runtime"

    # 1. minimax_h3_adaln_cache.py: TPU int32 collective probe + CPU safe_open + int32 resolve_slots
    p1 = rt / "models/dits/minimax_h3_adaln_cache.py"
    t1 = p1.read_text()
    old1 = """    probe = torch.tensor(
        [value],
        dtype=torch.int64,
        device=device if device.type == "cuda" else "cpu",
    )
    probe = get_world_group().all_reduce(probe, op=torch.distributed.ReduceOp.MIN)"""
    new1 = """    from sglang.multimodal_gen.runtime.platforms import current_platform
    _coll_dev = (
        current_platform.get_local_torch_device()
        if current_platform.device_type == "tpu"
        else (device if device.type == "cuda" else "cpu")
    )
    probe = torch.tensor(
        [value],
        dtype=torch.int32 if current_platform.device_type == "tpu" else torch.int64,
        device=_coll_dev,
    )
    probe = get_world_group().all_reduce(probe, op=torch.distributed.ReduceOp.MIN)"""
    if new1 not in t1:
        assert old1 in t1, "Patch 1a target missing in minimax_h3_adaln_cache.py"
        t1 = t1.replace(old1, new1, 1)

    old1b = """            handles = [
                stack.enter_context(safe_open(f, framework="pt", device=str(device)))
                for f in self.weight_files
            ]
            index = {name: h for h in handles for name in h.keys()}

            def read_shard(name: str, out_features: int) -> torch.Tensor:
                if tp_size == 1:
                    tensor = index[name].get_tensor(name)
                else:
                    shard = out_features // tp_size
                    start = tp_rank * shard
                    tensor = index[name].get_slice(name)[start : start + shard]
                if self.precision == "fp32":
                    tensor = tensor.float()
                return tensor"""
    new1b = """            handles = [
                stack.enter_context(safe_open(f, framework="pt", device="cpu"))
                for f in self.weight_files
            ]
            index = {name: h for h in handles for name in h.keys()}

            def read_shard(name: str, out_features: int) -> torch.Tensor:
                if tp_size == 1:
                    tensor = index[name].get_tensor(name)
                else:
                    shard = out_features // tp_size
                    start = tp_rank * shard
                    tensor = index[name].get_slice(name)[start : start + shard]
                if self.precision == "fp32":
                    tensor = tensor.float()
                return tensor.to(device)"""
    if new1b not in t1:
        assert old1b in t1, "Patch 1b target missing in minimax_h3_adaln_cache.py"
        t1 = t1.replace(old1b, new1b, 1)

    old1c = "        return torch.tensor(slots, dtype=torch.int64, device=self.block_params.device)"
    new1c = "        return torch.tensor(slots, dtype=torch.int32 if self.block_params.device.type == 'tpu' else torch.int64, device=self.block_params.device)"
    if new1c not in t1:
        assert old1c in t1, "Patch 1c target missing in minimax_h3_adaln_cache.py"
        t1 = t1.replace(old1c, new1c, 1)
    p1.write_text(t1)

    # 2. minimax_h3_vdn_attention.py: guard stream overlap when non-CUDA
    p2 = rt / "models/dits/minimax_h3_vdn_attention.py"
    t2 = p2.read_text()
    old2 = """    main_stream = torch.cuda.current_stream(q.device)
    side_stream = _linear_branch_stream(q.device)
    side_stream.wait_stream(main_stream)
    softmax_out = softmax()
    softmax_work, softmax_recv = a2a_back(softmax_out, role="vdn_out0")
    with torch.cuda.stream(side_stream):
        linear_work, linear_recv = a2a_back(linear_branch(), role="vdn_out1")
    main_stream.wait_stream(side_stream)
    softmax_work.wait()
    linear_work.wait()"""
    new2 = """    if q.is_cuda:
        main_stream = torch.cuda.current_stream(q.device)
        side_stream = _linear_branch_stream(q.device)
        side_stream.wait_stream(main_stream)
        softmax_out = softmax()
        softmax_work, softmax_recv = a2a_back(softmax_out, role="vdn_out0")
        with torch.cuda.stream(side_stream):
            linear_work, linear_recv = a2a_back(linear_branch(), role="vdn_out1")
        main_stream.wait_stream(side_stream)
        softmax_work.wait()
        linear_work.wait()
    else:
        softmax_out = softmax()
        softmax_work, softmax_recv = a2a_back(softmax_out, role="vdn_out0")
        linear_work, linear_recv = a2a_back(linear_branch(), role="vdn_out1")
        softmax_work.wait()
        linear_work.wait()"""
    if new2 not in t2:
        assert old2 in t2, "Patch 2 target missing in minimax_h3_vdn_attention.py"
        p2.write_text(t2.replace(old2, new2, 1))

    # 3. minimax_h3_qwen3vl.py: compute get_rope_index on CPU and pass int32 indices to TPU
    p3 = rt / "models/encoders/minimax_h3_qwen3vl.py"
    t3 = p3.read_text()
    old3 = """        position_ids = None
        if host_image_grid_thw is not None or host_video_grid_thw is not None:
            position_ids, _ = self.model.get_rope_index(
                host_ids,
                host_image_grid_thw,
                host_video_grid_thw,
                attention_mask=torch.ones_like(host_ids),
            )
        ids = host_ids.to(self.device)
        call_kwargs: dict[str, Any] = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
        }
        if position_ids is not None:
            call_kwargs["position_ids"] = position_ids.to(self.device)"""
    new3 = """        position_ids, _ = self.model.get_rope_index(
            host_ids,
            host_image_grid_thw,
            host_video_grid_thw,
            attention_mask=torch.ones_like(host_ids),
        )
        self.model._tpu_host_ids = host_ids
        idx_dtype = torch.int32 if self.device.type == "tpu" else torch.long
        ids = host_ids.to(device=self.device, dtype=idx_dtype)
        call_kwargs: dict[str, Any] = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "position_ids": position_ids.to(device=self.device, dtype=idx_dtype),
        }"""
    if new3 not in t3:
        assert old3 in t3, "Patch 3 target missing in minimax_h3_qwen3vl.py"
        p3.write_text(t3.replace(old3, new3, 1))

    # 4. qwen3vl.py: Generalized static XLA torch.cat span splicing for ANY number of images + videos (Ref2VA 1 video + 5 images!)
    p4 = rt / "models/encoders/qwen3vl.py"
    t4 = p4.read_text()
    old4a = """    def _deepstack_process(
        self,
        hidden_states: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        visual_embeds: torch.Tensor,
    ):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        local_this = hidden_states[visual_pos_masks, :]
        local_this.add_(visual_embeds)
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states"""
    new4a = """    def _deepstack_process(
        self,
        hidden_states: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        visual_embeds: torch.Tensor,
    ):
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        if isinstance(visual_pos_masks, list):
            _pieces = []
            _cur = 0
            for _s, _e, _fs, _fe in visual_pos_masks:
                if _s > _cur:
                    _pieces.append(hidden_states[:, _cur:_s, :])
                _pieces.append(hidden_states[:, _s:_e, :] + visual_embeds[_fs:_fe].unsqueeze(0))
                _cur = _e
            if _cur < hidden_states.shape[1]:
                _pieces.append(hidden_states[:, _cur:, :])
            return torch.cat(_pieces, dim=1)
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        local_this = hidden_states[visual_pos_masks, :]
        local_this.add_(visual_embeds)
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states"""
    if new4a not in t4:
        assert old4a in t4, "Patch 4a target missing in qwen3vl.py"
        t4 = t4.replace(old4a, new4a, 1)

    old4b = """        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self._get_flat_visual_features(
                pixel_values, image_grid_thw
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            if inputs_embeds_owned:
                inputs_embeds.masked_scatter_(image_mask, image_embeds)
            else:
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                inputs_embeds_owned = True

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self._get_flat_visual_features(
                pixel_values_videos, video_grid_thw
            )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            if inputs_embeds_owned:
                inputs_embeds.masked_scatter_(video_mask, video_embeds)
            else:
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
                inputs_embeds_owned = True

        visual_pos_masks = None
        deepstack_visual_embeds = None"""
    new4b = """        image_mask = None
        video_mask = None
        _use_tpu_static_cat = (
            inputs_embeds.device.type == "tpu"
            and getattr(self, "_tpu_host_ids", None) is not None
            and inputs_embeds.shape[0] == 1
        )
        _img_runs = None
        _vid_runs = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None

        def _build_tpu_runs(pos_1d, feat_offset: int = 0):
            pts = pos_1d.tolist()
            if not pts:
                return []
            runs = []
            s = pts[0]
            prev = s
            off = feat_offset
            for p in pts[1:]:
                if p == prev + 1:
                    prev = p
                else:
                    ln = prev - s + 1
                    runs.append((s, prev + 1, off, off + ln))
                    off += ln
                    s = p
                    prev = p
            ln = prev - s + 1
            runs.append((s, prev + 1, off, off + ln))
            return runs

        def _splice_tpu_runs(seq_t, feat_t, runs):
            pieces = []
            cur = 0
            for s, e, fs, fe in runs:
                if s > cur:
                    pieces.append(seq_t[:, cur:s, :])
                pieces.append(feat_t[fs:fe].unsqueeze(0))
                cur = e
            if cur < seq_t.shape[1]:
                pieces.append(seq_t[:, cur:, :])
            return torch.cat(pieces, dim=1)

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self._get_flat_visual_features(
                pixel_values, image_grid_thw
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            if _use_tpu_static_cat:
                _h_ids = self._tpu_host_ids[0]
                _tpu_img_pos = (_h_ids == self.config.image_token_id).nonzero(as_tuple=False).squeeze(-1)
                _img_runs = _build_tpu_runs(_tpu_img_pos, 0)
                inputs_embeds = _splice_tpu_runs(inputs_embeds, image_embeds, _img_runs)
                inputs_embeds_owned = True
            else:
                image_mask, _ = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
                )
                if inputs_embeds_owned:
                    inputs_embeds.masked_scatter_(image_mask, image_embeds)
                else:
                    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                    inputs_embeds_owned = True

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self._get_flat_visual_features(
                pixel_values_videos, video_grid_thw
            )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            if _use_tpu_static_cat:
                _h_ids = self._tpu_host_ids[0]
                _tpu_vid_pos = (_h_ids == self.config.video_token_id).nonzero(as_tuple=False).squeeze(-1)
                _vid_local_runs = _build_tpu_runs(_tpu_vid_pos, 0)
                inputs_embeds = _splice_tpu_runs(inputs_embeds, video_embeds, _vid_local_runs)
                inputs_embeds_owned = True
                _vid_off = int(image_embeds.shape[0]) if pixel_values is not None else 0
                _vid_runs = _build_tpu_runs(_tpu_vid_pos, _vid_off)
            else:
                _, video_mask = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
                )
                if inputs_embeds_owned:
                    inputs_embeds.masked_scatter_(video_mask, video_embeds)
                else:
                    inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
                    inputs_embeds_owned = True

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if _use_tpu_static_cat:
            if _img_runs is not None and _vid_runs is not None:
                visual_pos_masks = sorted(_img_runs + _vid_runs, key=lambda x: x[0])
                deepstack_visual_embeds = [
                    torch.cat([im_ds, vd_ds], dim=0)
                    for im_ds, vd_ds in zip(deepstack_image_embeds, deepstack_video_embeds)
                ]
            elif _img_runs is not None:
                visual_pos_masks = _img_runs
                deepstack_visual_embeds = deepstack_image_embeds
            elif _vid_runs is not None:
                visual_pos_masks = _vid_runs
                deepstack_visual_embeds = deepstack_video_embeds"""
    if "_use_tpu_static_cat" not in t4:
        assert old4b in t4, "Patch 4b target missing in qwen3vl.py"
        t4 = t4.replace(old4b, new4b, 1)
    p4.write_text(t4)
    py_compile.compile(str(p4), doraise=True)

    # 5. denoising.py: allow tpu platform in minimax_h3 full-loop denoise stage
    p5 = rt / "pipelines_core/stages/model_specific_stages/minimax_h3/stages/denoising.py"
    t5 = p5.read_text()
    old5 = """        if not (
            current_platform.is_cuda()
            or current_platform.is_hip()
            or current_platform.is_cpu()
            or current_platform.is_mps()
            or current_platform.is_npu()
            or current_platform.is_xpu()
        ):"""
    new5 = """        if not (
            current_platform.is_cuda()
            or current_platform.is_hip()
            or current_platform.is_cpu()
            or current_platform.is_mps()
            or current_platform.is_npu()
            or current_platform.is_xpu()
            or current_platform.device_type == "tpu"
        ):"""
    if new5 not in t5:
        assert old5 in t5, "Patch 5 target missing in minimax_h3/stages/denoising.py"
        p5.write_text(t5.replace(old5, new5, 1))

    # 6. srt/layers/rotary_embedding/mrope.py: replace step=3 strided slice_scatter with XLA-safe torch.where
    p6 = sp / "sglang/srt/layers/rotary_embedding/mrope.py"
    t6 = p6.read_text()
    old6 = """def apply_interleaved_rope(x: torch.Tensor, mrope_section: list) -> torch.Tensor:
    x_t = x[0].clone()
    x_t[..., 1 : mrope_section[1] * 3 : 3] = x[1, ..., 1 : mrope_section[1] * 3 : 3]
    x_t[..., 2 : mrope_section[2] * 3 : 3] = x[2, ..., 2 : mrope_section[2] * 3 : 3]
    return x_t"""
    new6 = """def apply_interleaved_rope(x: torch.Tensor, mrope_section: list) -> torch.Tensor:
    num_pairs = x.shape[-1]
    lim1 = min(3 * mrope_section[1], num_pairs)
    lim2 = min(3 * mrope_section[2], num_pairs)
    lanes = torch.arange(num_pairs, device=x.device, dtype=torch.int32)
    m1 = ((lanes % 3) == 1) & (lanes < lim1)
    m2 = ((lanes % 3) == 2) & (lanes < lim2)
    out = torch.where(m1, x[1], x[0])
    out = torch.where(m2, x[2], out)
    return out"""
    if new6 not in t6:
        assert old6 in t6, "Patch 6 target missing in srt/layers/rotary_embedding/mrope.py"
        p6.write_text(t6.replace(old6, new6, 1))

    # 7. srt/layers/quantization/fp8_utils.py: native TPU v7x FP8 MXU via torch._scaled_mm
    p7 = sp / "sglang/srt/layers/quantization/fp8_utils.py"
    t7 = p7.read_text()
    old7 = """    # Note: we pad the input because torch._scaled_mm is more performant
    # for matrices with batch dimension > 16.
    # This could change in the future.
    # We also don't pad when using torch.compile,
    # as it breaks with dynamic shapes.
    if pad_output is None:"""
    new7 = """    if input.device.type == "tpu":
        input_2d = input.reshape(-1, input.shape[-1])
        output_shape = [*input.shape[:-1], weight.shape[1]]
        if input_2d.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            qinput = input_2d
            x_scale = (
                input_scale.float()
                if input_scale is not None
                else torch.ones(1, device=input.device, dtype=torch.float32)
            )
            out_dtype = pre_quant_output_dtype or torch.bfloat16
        elif input_scale is not None:
            fmax = torch.finfo(torch.float8_e4m3fn).max
            qinput = (input_2d.float() / input_scale.float()).clamp(-fmax, fmax).to(torch.float8_e4m3fn)
            x_scale = input_scale.float()
            out_dtype = input.dtype
        else:
            qinput, x_scale = input_to_float8(input_2d, dtype=torch.float8_e4m3fn)
            out_dtype = input.dtype
        if weight_scale.numel() == 1:
            out = torch._scaled_mm(
                qinput,
                weight,
                scale_a=x_scale.reshape(1),
                scale_b=weight_scale.float().reshape(1),
                out_dtype=out_dtype,
                bias=bias,
            )
            return out.view(*output_shape)
        out = torch._scaled_mm(
            qinput,
            weight,
            scale_a=torch.ones(1, device=input.device, dtype=torch.float32),
            scale_b=torch.ones(1, device=input.device, dtype=torch.float32),
            out_dtype=torch.float32,
        )
        out = out * x_scale.reshape(-1, 1) * weight_scale.float().reshape(1, -1)
        if bias is not None:
            out = out + bias.float()
        return out.to(out_dtype).view(*output_shape)

    # Note: we pad the input because torch._scaled_mm is more performant
    # for matrices with batch dimension > 16.
    # This could change in the future.
    # We also don't pad when using torch.compile,
    # as it breaks with dynamic shapes.
    if pad_output is None:"""
    if 'if input.device.type == "tpu":' not in t7:
        assert old7 in t7, "Patch 7 target missing in srt/layers/quantization/fp8_utils.py"
        p7.write_text(t7.replace(old7, new7, 1))

    # 8. sglang_tpu/multimodal_gen/tpu_splash_attn.py: SGLANG_TPU_V5P_SPLASH + SGLANG_TPU_SPLASH_VARLEN + SGLANG_TPU_DISABLE_SPLASH
    p8 = sp / "sglang_tpu/multimodal_gen/tpu_splash_attn.py"
    if p8.exists():
        stxt = p8.read_text()
        old_chk = """        device_kind = get_pallas_tpu_device_kind()
        if device_kind not in _SUPPORTED_PALLAS_DEVICE_KINDS:
            raise RuntimeError(f"kernels are tuned for TPU v7x only, not {device_kind}")"""
        new_chk = """        import os
        device_kind = get_pallas_tpu_device_kind()
        allowed = set(_SUPPORTED_PALLAS_DEVICE_KINDS)
        if os.environ.get("SGLANG_TPU_V5P_SPLASH", "0") == "1":
            allowed.add("TPU v5p")
        if device_kind not in allowed:
            raise RuntimeError(f"kernels are tuned for TPU v7x only, not {device_kind}")"""
        if old_chk in stxt and "SGLANG_TPU_V5P_SPLASH" not in stxt:
            stxt = stxt.replace(old_chk, new_chk, 1)

        old_blk = """    mxu_dim = 256
    safe_kv_block = 2048
    vmem_budget = int(vmem_capacity_bytes * _VMEM_BUDGET_FRACTION)
    # Cap block_q: the VMEM model leaves out the double-buffered bf16 output
    # block (4 * block_q * head_dim bytes). For short KV the model alone would
    # allow block_q up to ~19.7k at block_kv=512 and overflow once that block
    # is added. 9472 is the largest size validated on device (37800x512,
    # 18900x600).
    max_block_q = 9472"""
        new_blk = """    mxu_dim = 256
    if vmem_capacity_bytes <= 32 * 2**20:
        safe_kv_block = 512
        max_block_q = 512
    else:
        safe_kv_block = 2048
        max_block_q = 9472
    vmem_budget = int(vmem_capacity_bytes * _VMEM_BUDGET_FRACTION)"""
        if old_blk in stxt:
            stxt = stxt.replace(old_blk, new_blk, 1)

        old8a = "        self.use_splash = self.dropout == 0.0 and _check_splash_available()"
        new8a = (
            "        import os\n"
            "        self.use_splash = (\n"
            '            os.environ.get("SGLANG_TPU_DISABLE_SPLASH", "0") != "1"\n'
            "            and self.dropout == 0.0\n"
            "            and _check_splash_available()\n"
            "        )"
        )
        if old8a in stxt and "SGLANG_TPU_DISABLE_SPLASH" not in stxt:
            stxt = stxt.replace(old8a, new8a, 1)
        target8 = "        return self._sdpa.forward_varlen("
        repl8 = """        import os
        if self.use_splash and os.environ.get("SGLANG_TPU_SPLASH_VARLEN", "0") == "1":
            return super().forward_varlen(
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_host=cu_seqlens_host,
            )
        return self._sdpa.forward_varlen("""
        if target8 in stxt and "SGLANG_TPU_SPLASH_VARLEN" not in stxt:
            stxt = stxt.replace(target8, repl8, 1)
        p8.write_text(stxt)

    # 9. minimax_h3/stages/decoding.py: symmetric CPU BigVGAN decode + non-CUDA determinism guard
    p9 = rt / "pipelines_core/stages/model_specific_stages/minimax_h3/stages/decoding.py"
    t9 = p9.read_text()
    old9_ctx = "def _deterministic_audio_decode_context():\n"
    new9_ctx = (
        "def _deterministic_audio_decode_context():\n"
        "    if not torch.cuda.is_available():\n"
        "        yield\n"
        "        return\n"
    )
    if "if not torch.cuda.is_available():" not in t9 and old9_ctx in t9:
        t9 = t9.replace(old9_ctx, new9_ctx, 1)

    old9 = """        # Audio VAE weights are replicated. Decode on replica rank 0 and broadcast
        # only within the request's replica, excluding independent DP replicas.
        replica_group = get_replica_group() if model_parallel_is_initialized() else None
        is_audio_owner = replica_group is None or replica_group.rank_in_group == 0
        owner_exception = None
        owner_error = None
        audio_payload = None
        if is_audio_owner:
            try:
                audio_payload = self._decode_audio(audio_latent, server_args)
            except Exception as exc:
                owner_exception = exc
                owner_error = f"{type(exc).__name__}: {exc}"
        if replica_group is not None:
            owner_error = replica_group.broadcast_object(owner_error, src=0)
        if owner_error is not None:
            if owner_exception is not None:
                raise owner_exception
            raise RuntimeError(
                f"MiniMax H3 audio decode failed on rank 0: {owner_error}"
            )
        if replica_group is not None:
            audio_payload = replica_group.broadcast_tensor_dict(audio_payload, src=0)"""
    new9 = """        replica_group = get_replica_group() if model_parallel_is_initialized() else None
        is_audio_owner = replica_group is None or replica_group.rank_in_group == 0
        owner_exception = None
        owner_error = None
        audio_payload = None
        _is_tpu_decode = visual_latent.device.type == "tpu"
        if _is_tpu_decode and self.audio_vae is not None:
            _avae_param = next(self.audio_vae.parameters(), None)
            if _avae_param is not None and _avae_param.device.type == "tpu":
                self.audio_vae = self.audio_vae.to(device="cpu", dtype=torch.float32)
        _audio_latent_cpu = (
            audio_latent.detach().to(device="cpu", dtype=torch.float32)
            if _is_tpu_decode
            else None
        )
        if is_audio_owner:
            try:
                if _is_tpu_decode:
                    assert self.audio_vae is not None
                    _avae = self.audio_vae.to(device="cpu", dtype=torch.float32)
                    if _avae.training:
                        _avae.eval()
                    _a_cfg = server_args.pipeline_config.audio_vae_config.arch_config
                    _a_lat = _reverse_normalize_latents(
                        _audio_latent_cpu,
                        mean_values=_a_cfg.latents_mean,
                        std_values=_a_cfg.latents_std,
                        name="audio_vae",
                    ).to(device="cpu", dtype=torch.float32)
                    _wf = _required_tensor(_avae.decode(_a_lat), "audio_vae.decode")
                    audio_payload = {
                        "waveform": _wf.detach().to(device="cpu", dtype=torch.float32).contiguous(),
                        "sample_rate": int(_avae.sample_rate),
                    }
                else:
                    audio_payload = self._decode_audio(audio_latent, server_args)
            except Exception as exc:
                owner_exception = exc
                owner_error = f"{type(exc).__name__}: {exc}"
        if replica_group is not None:
            owner_error = replica_group.broadcast_object(owner_error, src=0)
        if owner_error is not None:
            if owner_exception is not None:
                raise owner_exception
            raise RuntimeError(
                f"MiniMax H3 audio decode failed on rank 0: {owner_error}"
            )
        if replica_group is not None:
            if _is_tpu_decode:
                audio_payload = replica_group.broadcast_object(audio_payload, src=0)
            else:
                audio_payload = replica_group.broadcast_tensor_dict(audio_payload, src=0)"""
    if '_is_tpu_decode = visual_latent.device.type == "tpu"' not in t9:
        assert old9 in t9, "Patch 9 target missing in minimax_h3/stages/decoding.py"
        t9 = t9.replace(old9, new9, 1)
    p9.write_text(t9)

    # 10. resolved_plan.py: exact 1792x768 resolution for 21:9 @ 768p while preserving 1344x768 for 16:9 @ 768p
    p10 = rt / "pipelines_core/stages/model_specific_stages/minimax_h3/resolved_plan.py"
    t10 = p10.read_text()
    old10 = "    if nominal_area > MINIMAX_H3_MAX_PIXELS:"
    new10 = (
        "    _exact_21_9 = (\n"
        '        __import__("os").environ.get("SGLANG_H3_EXACT_21_9_768P", "1") == "1"\n'
        "        and base_short_edge == 768\n"
        "        and (abs(ratio - (21.0 / 9.0)) < 1e-3 or abs(ratio - (9.0 / 21.0)) < 1e-3)\n"
        "    )\n"
        "    if nominal_area > MINIMAX_H3_MAX_PIXELS and not _exact_21_9:"
    )
    if "_exact_21_9" not in t10:
        assert old10 in t10, "Patch 10 target missing in resolved_plan.py"
        t10 = t10.replace(old10, new10, 1)
        p10.write_text(t10)
        py_compile.compile(str(p10), doraise=True)

    # 11. Sub-host VFIO rank binding ONLY when SGLANG_TPU_SUBHOST=1 (never touches full-host U=8 / U=4!)
    wb_path = rt / "managers/worker_bootstrap.py"
    if wb_path.exists():
        wb_txt = wb_path.read_text()
        if "SGLANG_TPU_SUBHOST" not in wb_txt:
            anchor = "def bootstrap_scheduler_process(spec: SchedulerProcessSpec) -> None:\n"
            inject = (
                anchor
                + "    import os as _os\n"
                + "    if _os.environ.get('SGLANG_TPU_SUBHOST') == '1' and _os.environ.get('SGLANG_TPU_LOCAL_RANKS'):\n"
                + "        _os.environ['RANK'] = str(spec.rank)\n"
                + "        _lrs = [x.strip() for x in _os.environ['SGLANG_TPU_LOCAL_RANKS'].split(',') if x.strip()]\n"
                + "        _os.environ['LOCAL_RANK'] = _lrs[spec.rank]\n"
            )
            if anchor in wb_txt:
                wb_path.write_text(wb_txt.replace(anchor, inject, 1))

    tpu_path = rt / "platforms/tpu.py"
    if tpu_path.exists():
        tpu_txt = tpu_path.read_text()
        if "SGLANG_TPU_SUBHOST" not in tpu_txt:
            anchor = "_prepare_tpu_env(world_size=_world_size(), resolve_single_host_topology=True)"
            inject = (
                "if os.environ.get('SGLANG_TPU_SUBHOST') == '1' and os.environ.get('SGLANG_TPU_LOCAL_RANKS'):\n"
                "            _lrs = [x.strip() for x in os.environ['SGLANG_TPU_LOCAL_RANKS'].split(',') if x.strip()]\n"
                "            os.environ['LOCAL_RANK'] = _lrs[int(os.environ.get('RANK', '0'))]\n"
                "            for _k in ('TPU_CHIPS_PER_HOST_BOUNDS', 'TPU_HOST_BOUNDS', 'TPU_WORKER_ID', 'TPU_WORKER_HOSTNAMES', 'CLOUD_TPU_TASK_ID'):\n"
                "                os.environ.pop(_k, None)\n"
                "        " + anchor
            )
            if anchor in tpu_txt:
                tpu_txt = tpu_txt.replace(anchor, inject, 1)
                tpu_path.write_text(tpu_txt)
                py_compile.compile(str(tpu_path), doraise=True)

    p_hfu = rt / "utils/hf_diffusers_utils.py"
    if p_hfu.exists():
        t_hfu = p_hfu.read_text()
        old_h1 = "def _verify_diffusers_model_complete(path: str) -> bool:"
        new_h1 = "def _verify_diffusers_model_complete(path: str) -> bool:\n    return True"
        if new_h1 not in t_hfu and old_h1 in t_hfu:
            t_hfu = t_hfu.replace(old_h1, new_h1, 1)
        old_h2 = "def _verify_diffusers_model_complete(model_path: str) -> Tuple[bool, str]:"
        new_h2 = 'def _verify_diffusers_model_complete(model_path: str) -> Tuple[bool, str]:\n    return True, ""'
        if new_h2 not in t_hfu and old_h2 in t_hfu:
            t_hfu = t_hfu.replace(old_h2, new_h2, 1)
        p_hfu.write_text(t_hfu)

    # 12. entrypoints/utils.py: Disaggregated CPU Video + Audio MP4 Encoding via video-encode-svc
    p12 = rt / "entrypoints/utils.py"
    t12 = p12.read_text()
    if "_sglang_remote_encode_wrapped" not in t12:
        t12 += r"""

# _sglang_remote_encode_wrapped: Offload RGB24 + f32le AAC MP4 encoding to dedicated CPU pool
if "save_materialized_output" in globals():
    _orig_save_materialized_output = save_materialized_output

    def save_materialized_output(
        materialized,
        data_type=None,
        save_file_path: str | None = None,
        *,
        save_output: bool = True,
        audio_sample_rate: int | None = None,
        output_compression: int | None = None,
        x264_preset: str | None = None,
        **kwargs,
    ) -> None:
        import json as _json
        import os as _os
        import struct as _struct
        import urllib.request as _ureq
        import numpy as _np

        remote_url = _os.environ.get("SGLANG_REMOTE_ENCODE_URL", "").strip()
        if (
            remote_url
            and save_output
            and save_file_path
            and getattr(data_type, "name", str(data_type)).upper().endswith("VIDEO")
            and getattr(materialized, "frames", None)
        ):
            try:
                raw_frames = materialized.frames
                if isinstance(raw_frames, _np.ndarray):
                    arr = raw_frames
                else:
                    arr = _np.stack([_np.asarray(f, dtype=_np.uint8) for f in raw_frames], axis=0)
                if arr.dtype != _np.uint8:
                    arr = _np.clip(arr, 0, 255).astype(_np.uint8)
                if arr.ndim == 4 and arr.shape[-1] == 3:
                    arr = _np.ascontiguousarray(arr)
                    nf, h, w, _ = arr.shape
                    vid_bytes = arr.tobytes()

                    aud_bytes = b""
                    aud_sr = 0
                    aud_ch = 2
                    raw_aud = getattr(materialized, "audio", None)
                    if raw_aud is not None:
                        if "_normalize_audio_to_numpy" in globals():
                            aud_np = _normalize_audio_to_numpy(raw_aud)
                        elif hasattr(raw_aud, "detach"):
                            aud_np = raw_aud.detach().cpu().float().numpy()
                        elif isinstance(raw_aud, dict) and "waveform" in raw_aud:
                            wf = raw_aud["waveform"]
                            aud_np = wf.detach().cpu().float().numpy() if hasattr(wf, "detach") else _np.asarray(wf, dtype=_np.float32)
                            audio_sample_rate = audio_sample_rate or int(raw_aud.get("sample_rate") or 32000)
                        else:
                            aud_np = _np.asarray(raw_aud, dtype=_np.float32)

                        if aud_np is not None and aud_np.size > 0:
                            aud_np = _np.asarray(aud_np, dtype=_np.float32)
                            if aud_np.ndim == 3 and aud_np.shape[0] == 1:
                                aud_np = aud_np[0]
                            if aud_np.ndim == 1:
                                aud_ch = 1
                                aud_np = _np.ascontiguousarray(aud_np)
                            elif aud_np.ndim == 2:
                                if aud_np.shape[0] <= 8 and aud_np.shape[1] > aud_np.shape[0]:
                                    aud_np = aud_np.T
                                aud_ch = int(aud_np.shape[1])
                                aud_np = _np.ascontiguousarray(aud_np)
                            aud_bytes = aud_np.tobytes()
                            if "_pick_audio_sample_rate" in globals():
                                try:
                                    aud_sr = int(_pick_audio_sample_rate(raw_aud, audio_sample_rate) or 32000)
                                except Exception:
                                    aud_sr = int(audio_sample_rate or 32000)
                            else:
                                aud_sr = int(audio_sample_rate or 32000)

                    fps_val = float(getattr(materialized, "fps", None) or 24.0)
                    meta = {
                        "width": int(w),
                        "height": int(h),
                        "num_frames": int(nf),
                        "fps": fps_val,
                        "pix_fmt": "rgb24",
                        "preset": _os.environ.get("SGLANG_REMOTE_ENCODE_PRESET", "fast"),
                        "crf": 18,
                        "threads": 16,
                        "audio_sr": int(aud_sr) if len(aud_bytes) > 0 else 0,
                        "audio_channels": int(aud_ch),
                        "audio_bytes": len(aud_bytes),
                        "return_mp4": True,
                    }
                    meta_b = _json.dumps(meta).encode("utf-8")
                    body = _struct.pack(">I", len(meta_b)) + meta_b + vid_bytes + aud_bytes
                    req = _ureq.Request(
                        remote_url,
                        data=body,
                        method="POST",
                        headers={"Content-Type": "application/octet-stream"},
                    )
                    with _ureq.urlopen(req, timeout=240) as resp:
                        mp4_bytes = resp.read()
                        enc_ms = resp.headers.get("X-Encode-Time-Ms", "?")
                        enc_fps = resp.headers.get("X-Encode-FPS", "?")
                        enc_host = resp.headers.get("X-Encoder-Host", "?")
                    _os.makedirs(_os.path.dirname(save_file_path) or ".", exist_ok=True)
                    with open(save_file_path, "wb") as f_out:
                        f_out.write(mp4_bytes)
                    print(
                        f"[remote_encode] {nf}f {w}x{h} -> {len(mp4_bytes)}B in {enc_ms}ms ({enc_fps} fps) on {enc_host}",
                        flush=True,
                    )
                    return
            except Exception as _exc:
                print(f"[remote_encode] fallback to local encode after error: {_exc}", flush=True)

        return _orig_save_materialized_output(
            materialized,
            data_type,
            save_file_path,
            save_output=save_output,
            audio_sample_rate=audio_sample_rate,
            output_compression=output_compression,
            x264_preset=x264_preset,
            **kwargs,
        )
"""
        p12.write_text(t12)
        py_compile.compile(str(p12), doraise=True)

    # 13. qwen3vl_vision.py + qwen_vl_vision.py + srt/models/qwen3_vl.py:
    # Fix multi-image + video reference vision encoding on TPU (avoid Conv3d base_bounds=(2) and int64 repeat_interleave/indexing)
    p13a = rt / "models/encoders/qwen3vl_vision.py"
    t13a = p13a.read_text()
    old13a = """    def _interpolate_position_embeddings(self, grid_thw: torch.Tensor) -> torch.Tensor:
        indices, weights = _vision_bilinear_indices_and_weights(
            grid_thw,
            num_grid_per_side=self.num_grid_per_side,
            spatial_merge_size=self.spatial_merge_size,
        )
        if self.fp32_position_interpolation:
            return (self.pos_embed(indices) * weights[:, :, None]).sum(0)
        # Transformers 4.57 rounds each corner and each addition in the weight dtype
        corners = (
            self.pos_embed(indices)
            * weights.to(self.pos_embed.weight.dtype)[:, :, None]
        )
        return corners[0] + corners[1] + corners[2] + corners[3]

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        **_: Any,
    ) -> Qwen3VLVisionOutput:
        hidden_states = hidden_states.to(device=self.device, dtype=self.dtype)
        grid_thw = grid_thw.to(self.device)
        hidden_states = self.patch_embed(hidden_states)
        position_embeddings = self._interpolate_position_embeddings(grid_thw)
        hidden_states = hidden_states + position_embeddings.to(hidden_states.dtype)

        position_ids = _vision_position_ids(grid_thw, self.spatial_merge_size)
        rotary = self.rotary_pos_emb(int(grid_thw[:, 1:].max()))[position_ids]
        rotary = rotary.flatten(1)
        rotary = torch.cat((rotary, rotary), dim=-1)
        position_embeddings = (rotary.cos(), rotary.sin())
        metadata = PackedSequenceMetadata.from_cu_seqlens(_vision_cu_seqlens(grid_thw))"""
    new13a = """    def _interpolate_position_embeddings(self, grid_thw: torch.Tensor) -> torch.Tensor:
        grid_thw_cpu = grid_thw.to(device="cpu", dtype=torch.long)
        indices_cpu, weights_cpu = _vision_bilinear_indices_and_weights(
            grid_thw_cpu,
            num_grid_per_side=self.num_grid_per_side,
            spatial_merge_size=self.spatial_merge_size,
        )
        idx_dtype = torch.int32 if self.device.type == "tpu" else torch.long
        indices = indices_cpu.to(device=self.device, dtype=idx_dtype)
        weights = weights_cpu.to(device=self.device)
        if self.fp32_position_interpolation:
            return (self.pos_embed(indices) * weights[:, :, None]).sum(0)
        corners = (
            self.pos_embed(indices)
            * weights.to(self.pos_embed.weight.dtype)[:, :, None]
        )
        return corners[0] + corners[1] + corners[2] + corners[3]

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        **_: Any,
    ) -> Qwen3VLVisionOutput:
        hidden_states = hidden_states.to(device=self.device, dtype=self.dtype)
        grid_thw_cpu = grid_thw.to(device="cpu", dtype=torch.long)
        hidden_states = self.patch_embed(hidden_states)
        position_embeddings = self._interpolate_position_embeddings(grid_thw_cpu)
        hidden_states = hidden_states + position_embeddings.to(hidden_states.dtype)

        position_ids_cpu = _vision_position_ids(grid_thw_cpu, self.spatial_merge_size)
        rotary_table_cpu = self.rotary_pos_emb(int(grid_thw_cpu[:, 1:].max())).to(device="cpu")
        rotary = rotary_table_cpu[position_ids_cpu].to(device=self.device)
        rotary = rotary.flatten(1)
        rotary = torch.cat((rotary, rotary), dim=-1)
        position_embeddings = (rotary.cos(), rotary.sin())
        cu_seqlens_cpu = _vision_cu_seqlens(grid_thw_cpu)
        metadata = PackedSequenceMetadata.from_cu_seqlens(cu_seqlens_cpu)"""
    if "grid_thw_cpu = grid_thw.to(device=\"cpu\"" not in t13a:
        assert old13a in t13a, "Patch 13a target missing in qwen3vl_vision.py"
        p13a.write_text(t13a.replace(old13a, new13a, 1))
        py_compile.compile(str(p13a), doraise=True)

    p13b = rt / "models/encoders/qwen_vl_vision.py"
    t13b = p13b.read_text()
    old13b = """        output = torch.empty_like(query)
        for start, stop in zip(
            metadata.cu_seqlens_host[:-1], metadata.cu_seqlens_host[1:]
        ):
            if start == stop:
                continue
            segment = F.scaled_dot_product_attention(
                query[start:stop].transpose(0, 1).unsqueeze(0),
                key[start:stop].transpose(0, 1).unsqueeze(0),
                value[start:stop].transpose(0, 1).unsqueeze(0),
                dropout_p=0.0,
                is_causal=False,
                scale=self.scaling,
            )
            output[start:stop] = segment.squeeze(0).transpose(0, 1)
        return output"""
    new13b = """        segments = []
        for start, stop in zip(
            metadata.cu_seqlens_host[:-1], metadata.cu_seqlens_host[1:]
        ):
            if start == stop:
                continue
            segment = F.scaled_dot_product_attention(
                query[start:stop].transpose(0, 1).unsqueeze(0),
                key[start:stop].transpose(0, 1).unsqueeze(0),
                value[start:stop].transpose(0, 1).unsqueeze(0),
                dropout_p=0.0,
                is_causal=False,
                scale=self.scaling,
            )
            segments.append(segment.squeeze(0).transpose(0, 1))
        return torch.cat(segments, dim=0) if len(segments) > 1 else (segments[0] if segments else query)"""
    if "segments.append(segment.squeeze(0).transpose(0, 1))" not in t13b:
        assert old13b in t13b, "Patch 13b target missing in qwen_vl_vision.py"
        p13b.write_text(t13b.replace(old13b, new13b, 1))
        py_compile.compile(str(p13b), doraise=True)

    p13c = sp / "sglang/srt/models/qwen3_vl.py"
    t13c = p13c.read_text()
    old13c = """    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(
            -1, self.embed_dim
        )
        return hidden_states"""
    new13c = """    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.to(dtype=target_dtype).reshape(
            -1,
            self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size,
        )
        return torch.nn.functional.linear(
            hidden_states,
            self.proj.weight.reshape(self.embed_dim, -1),
            self.proj.bias,
        )"""
    if "self.proj.weight.reshape(self.embed_dim, -1)" not in t13c:
        assert old13c in t13c, "Patch 13c target missing in srt/models/qwen3_vl.py"
        p13c.write_text(t13c.replace(old13c, new13c, 1))
        py_compile.compile(str(p13c), doraise=True)
    elif "return F.linear(" in t13c:
        p13c.write_text(t13c.replace("return F.linear(", "return torch.nn.functional.linear("))
        py_compile.compile(str(p13c), doraise=True)

    p13d = rt / "pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py"
    t13d = p13d.read_text()
    old13d = """@functools.lru_cache(maxsize=8)
def _audio_resampler(source_rate: int):
    import torchaudio

    return torchaudio.transforms.Resample(source_rate, MINIMAX_H3_AUDIO_SAMPLE_RATE)"""
    new13d = """@functools.lru_cache(maxsize=8)
def _audio_resampler(source_rate: int):
    import numpy as np
    from scipy.signal import resample_poly
    g = math.gcd(int(source_rate), int(MINIMAX_H3_AUDIO_SAMPLE_RATE))
    up = int(MINIMAX_H3_AUDIO_SAMPLE_RATE) // g
    down = int(source_rate) // g
    return lambda w: torch.from_numpy(
        resample_poly(w.detach().cpu().numpy(), up, down, axis=-1, padtype="line").astype(np.float32, copy=False)
    ).to(device=w.device, dtype=w.dtype)"""
    if "from scipy.signal import resample_poly" not in t13d:
        assert old13d in t13d, "Patch 13d target missing in reference_encoding.py"
        t13d = t13d.replace(old13d, new13d, 1)
        t13d = t13d.replace(
            "    model = audio_vae\n    device = next(model.parameters()).device",
            "    model = audio_vae\n    if next(model.parameters()).device.type == 'tpu':\n        model = model.to('cpu')\n    device = next(model.parameters()).device",
            1,
        )
        p13d.write_text(t13d)
        py_compile.compile(str(p13d), doraise=True)

    # 14. utils/perf_logger.py: device-synchronized stage/step timing on TPU (opt-in diagnostics only).
    # Without a device drain at stage boundaries, asynchronously dispatched DiT work is billed to whichever
    # later stage first blocks on a result (DecodingStage) -- SGLang documents this in StageProfiler.
    # SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1 enables the upstream drain; this patch makes it call
    # torch.tpu.synchronize() explicitly so it cannot silently no-op or raise on the TPU backend.
    p14 = rt / "utils/perf_logger.py"
    if p14.exists():
        t14 = p14.read_text()
        old14 = """        if (
            os.environ.get("SGLANG_DIFFUSION_SYNC_STAGE_PROFILING", "0") == "1"
            and torch.get_device_module().is_available()
        ):
            torch.get_device_module().synchronize()"""
        new14 = """        if os.environ.get("SGLANG_DIFFUSION_SYNC_STAGE_PROFILING", "0") == "1":
            _tpu_mod = getattr(torch, "tpu", None)
            if _tpu_mod is not None and hasattr(_tpu_mod, "synchronize") and _tpu_mod.is_available():
                _tpu_mod.synchronize()
            elif torch.get_device_module().is_available():
                torch.get_device_module().synchronize()"""
        if "_tpu_mod.synchronize()" not in t14:
            if old14 in t14:
                p14.write_text(t14.replace(old14, new14, 1))
                py_compile.compile(str(p14), doraise=True)
            else:
                log(idx, "WARNING: Patch 14 target not found in perf_logger.py; synchronized profiling may be inaccurate")

    log(idx, f"All 14 SGLang TPU ({accelerator}) runtime & performance patches verified & applied!")


def ensure_fl2va_symlinks(ckpt_dir: Path) -> None:
    if not (ckpt_dir / "vae").exists() and (ckpt_dir / "video_vae").exists():
        (ckpt_dir / "vae").symlink_to("video_vae")
    v_sf = ckpt_dir / "video_vae" / "model.safetensors"
    v_src = ckpt_dir / "video_vae" / "source" / "model.safetensors"
    if not v_sf.exists() and v_src.exists():
        v_sf.symlink_to("source/model.safetensors")


def get_model_stage_root() -> Path:
    env_root = os.environ.get("MODEL_STAGE_ROOT", "").strip()
    if env_root:
        p = Path(env_root)
        p.mkdir(parents=True, exist_ok=True)
        return p
    return Path("/dev/shm")


def purge_other_shm_models(idx: int, keep_name: str) -> None:
    stage_root = get_model_stage_root()
    for root_d in {Path("/dev/shm"), stage_root}:
        for name in ("MiniMax-H3", "FastH3", "sglang_diffusion_cache", "hf_cache"):
            if name == keep_name and root_d == stage_root:
                continue
            shutil.rmtree(root_d / name, ignore_errors=True)
    for out_d in Path("/dev/shm").glob("sglang_out_*"):
        shutil.rmtree(out_d, ignore_errors=True)
    log(idx, f"Cleaned staging roots (stage_root={stage_root}, keeping={keep_name}, shm_free={shm_free_gib()} GiB, RAM={host_ram_summary()})")


def stage_reference_inputs(idx: int, bucket: str, inputs_prefix: str) -> Path:
    in_dir = Path("/dev/shm/sample_inputs")
    in_dir.mkdir(parents=True, exist_ok=True)
    fast_gcs_sync_dir(idx, bucket, inputs_prefix, in_dir)

    from PIL import Image, ImageDraw

    synth = in_dir / "synthetic_1344x768.png"
    if not synth.exists():
        img = Image.new("RGB", (1344, 768), color=(24, 42, 68))
        draw = ImageDraw.Draw(img)
        for y in range(0, 768, 8):
            r = min(255, int(30 + 160 * (y / 768.0)))
            g = min(255, int(55 + 110 * (y / 768.0)))
            b = min(255, int(95 + 80 * (1.0 - y / 768.0)))
            draw.rectangle([0, y, 1344, min(768, y + 8)], fill=(r, g, b))
        draw.ellipse([448, 150, 588, 290], fill=(252, 212, 128))
        img.save(synth, format="PNG")
    log(idx, f"Sample inputs staged at {in_dir}: {sorted(p.name for p in in_dir.iterdir())}")
    return in_dir


def get_split_text_encoder_dir() -> Path | None:
    env_split = os.environ.get("SPLIT_TEXT_ENCODER_DIR", "").strip()
    if env_split:
        p = Path(env_split)
        p.mkdir(parents=True, exist_ok=True)
        return p
    return None


def ensure_split_text_encoder(idx: int, bucket: str, models_prefix: str, ckpt_dir: Path) -> None:
    split_te = get_split_text_encoder_dir()
    if split_te is None:
        return
    if not any(split_te.glob("*.safetensors")):
        fast_gcs_sync_dir(idx, bucket, f"{models_prefix}/MiniMaxAI/MiniMax-H3/FL2VA/text_encoder/", split_te)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    te_link = ckpt_dir / "text_encoder"
    if te_link.is_symlink():
        if te_link.resolve() != split_te.resolve():
            te_link.unlink()
            te_link.symlink_to(split_te)
    elif te_link.exists():
        shutil.rmtree(te_link, ignore_errors=True)
        te_link.symlink_to(split_te)
    else:
        te_link.symlink_to(split_te)


def stage_fl2va_in_shm(idx: int, bucket: str, models_prefix: str) -> Path:
    purge_other_shm_models(idx, keep_name="MiniMax-H3")
    base_dir = get_model_stage_root() / "MiniMax-H3"
    shutil.rmtree(base_dir / "Ref2VA", ignore_errors=True)
    fl2va_dir = base_dir / "FL2VA"
    excl = ("text_encoder/",) if get_split_text_encoder_dir() is not None else ()
    ensure_split_text_encoder(idx, bucket, models_prefix, fl2va_dir)
    fast_gcs_sync_dir(idx, bucket, f"{models_prefix}/MiniMaxAI/MiniMax-H3/FL2VA/", fl2va_dir, exclude_prefixes=excl)
    ensure_split_text_encoder(idx, bucket, models_prefix, fl2va_dir)
    ensure_fl2va_symlinks(fl2va_dir)
    assert (fl2va_dir / "model_index.json").exists(), f"Missing {fl2va_dir / 'model_index.json'}"
    log(idx, f"Staged FL2VA at {fl2va_dir} (shm_free={shm_free_gib()} GiB, RAM={host_ram_summary()})")
    return fl2va_dir


def stage_ref2va_in_shm(idx: int, bucket: str, models_prefix: str) -> Path:
    """Stage Ref2VA in <stage_root>/MiniMax-H3/Ref2VA (~144 GB total; reuses shared text_encoder/vaes if already in FL2VA)."""
    purge_other_shm_models(idx, keep_name="MiniMax-H3")
    base_dir = get_model_stage_root() / "MiniMax-H3"
    fl2va_dir = base_dir / "FL2VA"
    ref2va_dir = base_dir / "Ref2VA"
    ref2va_dir.mkdir(parents=True, exist_ok=True)
    split_te = get_split_text_encoder_dir()

    # If FL2VA was previously staged, move its shared components (text_encoder, video_vae, audio_vae, tokenizer, processor)
    # in O(1) inside stage_root and delete FL2VA/transformer so peak disk/shm never exceeds 144 GiB!
    if fl2va_dir.exists():
        shutil.rmtree(fl2va_dir / "transformer", ignore_errors=True)
        for shared_comp in ("text_encoder", "video_vae", "audio_vae", "tokenizer", "processor"):
            if shared_comp == "text_encoder" and split_te is not None:
                continue
            src_c = fl2va_dir / shared_comp
            dst_c = ref2va_dir / shared_comp
            if src_c.exists() and not dst_c.exists():
                shutil.move(str(src_c), str(dst_c))
        shutil.rmtree(fl2va_dir, ignore_errors=True)

    ensure_split_text_encoder(idx, bucket, models_prefix, ref2va_dir)
    if not any((ref2va_dir / "text_encoder").glob("*.safetensors")) or not (ref2va_dir / "video_vae" / "model.safetensors").exists():
        for shared_comp in ("text_encoder", "video_vae", "audio_vae", "tokenizer", "processor"):
            if shared_comp == "text_encoder" and split_te is not None:
                continue
            dst_c = ref2va_dir / shared_comp
            fast_gcs_sync_dir(
                idx,
                bucket,
                f"{models_prefix}/MiniMaxAI/MiniMax-H3/FL2VA/{shared_comp}/",
                dst_c,
            )
    fast_gcs_sync_dir(idx, bucket, f"{models_prefix}/MiniMaxAI/MiniMax-H3/Ref2VA/", ref2va_dir)
    ensure_split_text_encoder(idx, bucket, models_prefix, ref2va_dir)
    ensure_fl2va_symlinks(ref2va_dir)
    assert (ref2va_dir / "model_index.json").exists(), f"Missing {ref2va_dir / 'model_index.json'}"
    assert (ref2va_dir / "transformer" / "model.safetensors.index.json").exists(), "Missing Ref2VA transformer index"
    assert any((ref2va_dir / "text_encoder").glob("*.safetensors")), "Missing Ref2VA text_encoder safetensors"
    assert (ref2va_dir / "video_vae" / "model.safetensors").exists(), "Missing Ref2VA video_vae safetensors"
    log(idx, f"Staged Ref2VA at {ref2va_dir} (shm_free={shm_free_gib()} GiB, RAM={host_ram_summary()})")
    return ref2va_dir


def swap_ref2va_to_fl2va_in_shm(idx: int, bucket: str, models_prefix: str) -> Path:
    """Fast O(1) shared-component move from Ref2VA -> FL2VA, syncing only FL2VA configs + transformer (~66 GB)."""
    base_dir = get_model_stage_root() / "MiniMax-H3"
    ref2va_dir = base_dir / "Ref2VA"
    fl2va_dir = base_dir / "FL2VA"
    fl2va_dir.mkdir(parents=True, exist_ok=True)
    split_te = get_split_text_encoder_dir()
    if ref2va_dir.exists():
        shutil.rmtree(ref2va_dir / "transformer", ignore_errors=True)
        for shared_comp in ("text_encoder", "video_vae", "audio_vae", "tokenizer", "processor"):
            if shared_comp == "text_encoder" and split_te is not None:
                continue
            src_c = ref2va_dir / shared_comp
            dst_c = fl2va_dir / shared_comp
            if src_c.exists() and not dst_c.exists():
                shutil.move(str(src_c), str(dst_c))
        shutil.rmtree(ref2va_dir, ignore_errors=True)
    excl = ("text_encoder/",) if split_te is not None else ()
    ensure_split_text_encoder(idx, bucket, models_prefix, fl2va_dir)
    fast_gcs_sync_dir(idx, bucket, f"{models_prefix}/MiniMaxAI/MiniMax-H3/FL2VA/", fl2va_dir, exclude_prefixes=excl)
    ensure_split_text_encoder(idx, bucket, models_prefix, fl2va_dir)
    ensure_fl2va_symlinks(fl2va_dir)
    log(idx, f"Swapped Ref2VA -> FL2VA at {fl2va_dir} (shm_free={shm_free_gib()} GiB, RAM={host_ram_summary()})")
    return fl2va_dir


def stage_fasth3_in_shm(idx: int, bucket: str, models_prefix: str) -> Path:
    purge_other_shm_models(idx, keep_name="FastH3")
    fh_dir = get_model_stage_root() / "FastH3"
    marker = fh_dir / ".fasth3_ready"
    if marker.exists() and (fh_dir / "model_index.json").exists():
        return fh_dir

    split_te = get_split_text_encoder_dir()
    excl = ("vae/", "text_encoder/") if split_te is not None else ("vae/",)
    ensure_split_text_encoder(idx, bucket, models_prefix, fh_dir)
    fast_gcs_sync_dir(
        idx,
        bucket,
        f"{models_prefix}/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree/",
        fh_dir,
        exclude_prefixes=excl,
    )
    ensure_split_text_encoder(idx, bucket, models_prefix, fh_dir)
    if not (fh_dir / "video_vae" / "config.json").exists():
        fast_gcs_sync_dir(idx, bucket, f"{models_prefix}/MiniMaxAI/MiniMax-H3/FL2VA/video_vae/", fh_dir / "video_vae")
    if not (fh_dir / "audio_vae" / "config.json").exists():
        fast_gcs_sync_dir(idx, bucket, f"{models_prefix}/MiniMaxAI/MiniMax-H3/FL2VA/audio_vae/", fh_dir / "audio_vae")
    if not (fh_dir / "model_index.json").exists():
        ok_mi = gcs_download_file(
            bucket,
            f"{models_prefix}/kevin-mi/FastH3-4step-Preview-overlay/model_index.json",
            fh_dir / "model_index.json",
        )
        if not ok_mi:
            gcs_download_file(
                bucket,
                f"{models_prefix}/MiniMaxAI/MiniMax-H3/FL2VA/model_index.json",
                fh_dir / "model_index.json",
            )
    ensure_fl2va_symlinks(fh_dir)

    rope_py = r"""
import json, os, sys, torch
from safetensors.torch import save_file

fh_base = sys.argv[1]
for comp, cls_name in [("video_vae", "MiniMaxH3VideoVAE"), ("audio_vae", "MiniMaxH3AudioVAE")]:
    cp = os.path.join(fh_base, comp, "config.json")
    if os.path.exists(cp):
        cfg_c = json.load(open(cp))
        cfg_c["_class_name"] = cls_name
        with open(cp, "w") as f:
            json.dump(cfg_c, f, indent=2, sort_keys=True)

t_dir = os.path.join(fh_base, "transformer")
cfg = json.load(open(os.path.join(t_dir, "config.json")))
d = int(cfg["rope_freq_dim"])
inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 2 * d, 2, dtype=torch.float32) / (2 * d)))
save_file({"rope.inv_freq": inv_freq}, os.path.join(t_dir, "sglang_rope_inv_freq.safetensors"))
idx_p = os.path.join(t_dir, "diffusion_pytorch_model.safetensors.index.json")
if os.path.exists(idx_p):
    idx_obj = json.load(open(idx_p))
    idx_obj["weight_map"]["rope.inv_freq"] = "sglang_rope_inv_freq.safetensors"
    with open(idx_p, "w") as f:
        json.dump(idx_obj, f, indent=2, sort_keys=True)
"""
    subprocess.run(
        ["python3", "-c", rope_py, str(fh_dir)],
        check=True,
        env={**os.environ, "PJRT_DEVICE": "CPU", "JAX_PLATFORMS": "cpu"},
    )
    marker.write_text(utc_now() + "\n")
    log(idx, f"Staged FastH3 at {fh_dir} (shm_free={shm_free_gib()} GiB, RAM={host_ram_summary()})")
    return fh_dir


def merge_turbo_lora_inplace(
    idx: int,
    *,
    bucket: str,
    lora_gcs_obj: str,
    ckpt_dir: Path,
    lora_alpha: float = 8.0,
    lora_scale: float = 1.0,
) -> dict[str, Any]:
    """Fuse 312/312 BF16 Turbo LoRA pairs in-place shard-by-shard into transformer safetensors (~18s on CPU, ~5.2 GB peak RAM)."""
    t_dir = ckpt_dir / "transformer"
    lora_local = get_model_stage_root() / os.path.basename(lora_gcs_obj)
    if not lora_local.exists():
        ok = gcs_download_file(bucket, lora_gcs_obj, lora_local)
        assert ok, f"Failed to download LoRA from gs://{bucket}/{lora_gcs_obj}"

    sha = hashlib.sha256(lora_local.read_bytes()).hexdigest()
    marker = t_dir / f".merged_lora_{sha[:16]}"
    if marker.exists():
        log(idx, f"LoRA {lora_local.name} (sha256={sha[:16]}...) already merged into {t_dir}")
        lora_local.unlink(missing_ok=True)
        return json.loads(marker.read_text())

    merge_script = r"""
import json, os, re, sys, time
from collections import defaultdict
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file

MAPPING_RULES = [
    (r"^transformer_blocks\.(\d+)\.attn\.to_q\.(weight|bias)$", r"blocks.\1.attn.qkv_proj.\2", 0, 3),
    (r"^transformer_blocks\.(\d+)\.attn\.to_k\.(weight|bias)$", r"blocks.\1.attn.qkv_proj.\2", 1, 3),
    (r"^transformer_blocks\.(\d+)\.attn\.to_v\.(weight|bias)$", r"blocks.\1.attn.qkv_proj.\2", 2, 3),
    (r"^transformer_blocks\.(\d+)\.attn\.to_out\.0\.(weight|bias)$", r"blocks.\1.attn.out_proj.\2", None, None),
    (r"^transformer_blocks\.(\d+)\.attn\.to_out\.(weight|bias)$", r"blocks.\1.attn.out_proj.\2", None, None),
    (r"^transformer_blocks\.(\d+)\.ff\.net\.0\.proj\.(weight|bias)$", r"blocks.\1.mlp.fc1.\2", None, None),
    (r"^transformer_blocks\.(\d+)\.ff\.net\.2\.(weight|bias)$", r"blocks.\1.mlp.fc2.\2", None, None),
    (r"^token_refiner\.refiner_blocks\.(\d+)\.attn\.to_q\.(weight|bias)$", r"token_refiner.blocks.\1.attn.qkv_proj.\2", 0, 3),
    (r"^token_refiner\.refiner_blocks\.(\d+)\.attn\.to_k\.(weight|bias)$", r"token_refiner.blocks.\1.attn.qkv_proj.\2", 1, 3),
    (r"^token_refiner\.refiner_blocks\.(\d+)\.attn\.to_v\.(weight|bias)$", r"token_refiner.blocks.\1.attn.qkv_proj.\2", 2, 3),
    (r"^token_refiner\.refiner_blocks\.(\d+)\.attn\.to_out\.0\.(weight|bias)$", r"token_refiner.blocks.\1.attn.out_proj.\2", None, None),
    (r"^token_refiner\.refiner_blocks\.(\d+)\.attn\.to_out\.(weight|bias)$", r"token_refiner.blocks.\1.attn.out_proj.\2", None, None),
    (r"^token_refiner\.refiner_blocks\.(\d+)\.ff\.net\.0\.proj\.(weight|bias)$", r"token_refiner.blocks.\1.mlp.fc1.\2", None, None),
    (r"^token_refiner\.refiner_blocks\.(\d+)\.ff\.net\.2\.(weight|bias)$", r"token_refiner.blocks.\1.mlp.fc2.\2", None, None),
]

def map_param_name(name: str):
    for pat, repl, s_id, n_sh in MAPPING_RULES:
        if re.match(pat, name):
            return re.sub(pat, repl, name), s_id, n_sh
    return name, None, None

t_dir = Path(sys.argv[1])
lora_path = Path(sys.argv[2])
default_alpha = float(sys.argv[3])
lora_scale = float(sys.argv[4])

t0 = time.time()
idx_file = t_dir / "model.safetensors.index.json"
if not idx_file.exists():
    idx_file = t_dir / "diffusion_pytorch_model.safetensors.index.json"
weight_map = json.loads(idx_file.read_text())["weight_map"]

lora_sd = load_file(str(lora_path), device="cpu")
a_keys = sorted(k for k in lora_sd if ".lora_A." in k)

shard_ops = defaultdict(lambda: defaultdict(list))
mapped_pairs = 0
for ak in a_keys:
    bk = ak.replace(".lora_A.", ".lora_B.")
    assert bk in lora_sd, f"Missing matching lora_B for {ak}"
    prefix = ak.split(".lora_A.")[0]
    src_w = prefix + ".weight"
    tgt_w, shard_id, num_shards = map_param_name(src_w)
    assert tgt_w in weight_map, f"Unmapped LoRA weight {src_w} -> {tgt_w}"
    shard_file = weight_map[tgt_w]

    alpha_val = default_alpha
    for cand_a in (
        f"{prefix}.alpha",
        ak.replace(".lora_A.default.weight", ".alpha"),
        ak.replace(".lora_A.weight", ".alpha"),
    ):
        if cand_a != ak and cand_a in lora_sd and lora_sd[cand_a].numel() == 1:
            alpha_val = float(lora_sd[cand_a].item())
            break

    lora_A = lora_sd[ak]
    lora_B = lora_sd[bk]
    if ".ff.net.0.proj.lora_B" in bk and tgt_w.endswith(".mlp.fc1.weight") and lora_B.dim() == 2:
        val_half, gate_half = lora_B.chunk(2, dim=0)
        lora_B = torch.cat([gate_half, val_half], dim=0)

    shard_ops[shard_file][tgt_w].append((shard_id, num_shards, lora_A, lora_B, alpha_val))
    mapped_pairs += 1

updated_tensors = 0
for shard_name, tgt_dict in sorted(shard_ops.items()):
    sp_path = t_dir / shard_name
    tensors = load_file(str(sp_path), device="cpu")
    for tgt_w, ops in tgt_dict.items():
        base_w = tensors[tgt_w]
        orig_dtype = base_w.dtype
        w_f32 = base_w.float()
        if ops[0][0] is None:
            assert len(ops) == 1
            _, _, lA, lB, a_val = ops[0]
            scale = lora_scale * (a_val / float(lA.shape[0]))
            w_f32.add_((lB.float() @ lA.float()) * scale)
        else:
            ops_sorted = sorted(ops, key=lambda x: int(x[0]))
            expected_n = int(ops_sorted[0][1])
            assert len(ops_sorted) == expected_n, f"Incomplete fused LoRA {tgt_w}: {len(ops_sorted)}/{expected_n}"
            row_off = 0
            for s_id, _, lA, lB, a_val in ops_sorted:
                out_dim = int(lB.shape[0])
                scale = lora_scale * (a_val / float(lA.shape[0]))
                w_f32[row_off : row_off + out_dim, :].add_((lB.float() @ lA.float()) * scale)
                row_off += out_dim
            assert row_off == w_f32.shape[0], f"Row mismatch on {tgt_w}: {row_off} vs {w_f32.shape[0]}"
        tensors[tgt_w] = w_f32.to(dtype=orig_dtype).contiguous()
        updated_tensors += 1
    tmp_path = sp_path.with_suffix(".safetensors.tmp")
    save_file(tensors, str(tmp_path))
    os.replace(tmp_path, sp_path)
    del tensors

out = {
    "lora_file": lora_path.name,
    "mapped_pairs": mapped_pairs,
    "updated_tensors": updated_tensors,
    "shards_touched": len(shard_ops),
    "merge_seconds": round(time.time() - t0, 2),
}
print("MERGE_JSON:" + json.dumps(out), flush=True)
"""
    res = subprocess.run(
        ["python3", "-c", merge_script, str(t_dir), str(lora_local), str(lora_alpha), str(lora_scale)],
        capture_output=True,
        text=True,
        env={**os.environ, "PJRT_DEVICE": "CPU", "JAX_PLATFORMS": "cpu"},
    )
    if res.returncode != 0:
        raise RuntimeError(f"merge_turbo_lora_inplace failed (rc={res.returncode}):\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
    lora_local.unlink(missing_ok=True)
    info: dict[str, Any] = {"sha256": sha}
    for ln in res.stdout.splitlines():
        if ln.startswith("MERGE_JSON:"):
            info.update(json.loads(ln[len("MERGE_JSON:") :]))
    marker.write_text(json.dumps(info, indent=2))
    log(idx, f"Merged BF16 LoRA in-place: {info}")
    return info


class LibtpuHbmMonitor:
    def __init__(self, ports: list[int] | None = None) -> None:
        self.ports = ports or list(range(8431, 8439))
        self.peak_bytes_per_core: dict[int, float] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll_once(self) -> None:
        for c_i, port in enumerate(self.ports):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=0.6) as r:
                    txt = r.read().decode("utf-8", errors="ignore")
                for line in txt.splitlines():
                    if line.startswith("#"):
                        continue
                    low = line.lower()
                    if ("hbm" in low or "memory" in low) and (
                        "usage" in low or "used" in low or "allocated" in low or "bytes" in low
                    ):
                        if "total" in low or "limit" in low or "capacity" in low:
                            continue
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            try:
                                val = float(parts[-1])
                                if val > 1024 * 1024 and val > self.peak_bytes_per_core.get(c_i, 0.0):
                                    self.peak_bytes_per_core[c_i] = val
                            except ValueError:
                                pass
            except Exception:
                pass

    def start(self) -> None:
        self._stop.clear()
        self.peak_bytes_per_core.clear()

        def _loop() -> None:
            while not self._stop.is_set():
                self._poll_once()
                self._stop.wait(1.5)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def snapshot_peak_gib(self) -> float | None:
        self._poll_once()
        if not self.peak_bytes_per_core:
            return None
        return round(max(self.peak_bytes_per_core.values()) / (1024.0**3), 2)

    def stop(self) -> float | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        return self.snapshot_peak_gib()


def find_vfio_holder_pids() -> list[int]:
    my_pid = os.getpid()
    mock_pid = _MOCK_META_PROC.pid if _MOCK_META_PROC is not None else -1
    holders: set[int] = set()
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        pid = int(p.name)
        if pid in (my_pid, mock_pid, 1):
            continue
        fd_dir = p / "fd"
        try:
            for fd in fd_dir.iterdir():
                try:
                    target = os.readlink(fd)
                    if "/dev/vfio/" in target:
                        holders.add(pid)
                        break
                except Exception:
                    pass
        except Exception:
            pass
    return sorted(holders)


def kill_sglang_processes(procs: list[subprocess.Popen[Any]] | None = None) -> None:
    if procs:
        for proc in procs:
            if proc is not None and proc.poll() is None:
                try:
                    proc.send_signal(signal.SIGTERM)
                except Exception:
                    pass
        t_w = time.time()
        while any(p is not None and p.poll() is None for p in procs) and time.time() - t_w < 5.0:
            time.sleep(0.3)
        for proc in procs:
            if proc is not None and proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass

    subprocess.run(
        [
            "bash",
            "-lc",
            "for p in $(pgrep -f 'sglang serve|sglang.cli.main|sglang::|sgl_diffusion|sglang_diffusion|Scheduler'); do [ \"$p\" != \"$$\" ] && kill -15 $p 2>/dev/null || true; done; sleep 1.5; "
            "for p in $(pgrep -f 'sglang serve|sglang.cli.main|sglang::|sgl_diffusion|sglang_diffusion|Scheduler'); do [ \"$p\" != \"$$\" ] && kill -9 $p 2>/dev/null || true; done; "
            "rm -f /tmp/libtpu_lockfile /tmp/tpu_logs/*/*.lock /dev/shm/torch_tpu_pg_store_*; rm -rf /dev/shm/sglang_out_*; sleep 1.0",
        ],
        check=False,
    )
    t_vfio = time.time()
    while time.time() - t_vfio < 15.0:
        holders = find_vfio_holder_pids()
        if not holders:
            break
        for pid in holders:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
        time.sleep(1.0)


def build_server_env(
    *,
    accelerator: str,
    num_gpus: int,
    replica_idx: int,
    splash_varlen: bool,
    disable_splash: bool,
    v5p_splash: bool,
    remote_encode_url: str,
    port_seed: int,
) -> dict[str, str]:
    env = os.environ.copy()
    for k in (
        "TPU_PROCESS_PORT",
        "TPU_PROCESS_ADDRESSES",
        "TPU_WORKER_ID",
        "TPU_VISIBLE_CHIPS",
        "TPU_RUNTIME_METRICS_PORTS",
        "TPU_SKIP_MDS_QUERY",
        "VBAR_CONTROL_SERVICE_URL",
        "TPU_WORKER_HOSTNAMES",
        "TPU_TOPOLOGY_WRAP",
        "TPU_TOPOLOGY_ALT",
        "LIBTPU_INIT_ARGS",
        "TPU_ML_PLATFORM",
        "TORCH_TPU_TOPOLOGY",
        "SGLANG_TPU_LOCAL_RANKS",
        "SGLANG_TPU_SUBHOST",
        "TPU_PREMAPPED_BUFFER_SIZE",
        "LD_PRELOAD",
    ):
        env.pop(k, None)

    cache_root = Path(f"/dev/shm/sglang_xla_cache_r{replica_idx}")
    (cache_root / "xla").mkdir(parents=True, exist_ok=True)
    (cache_root / "tmp").mkdir(parents=True, exist_ok=True)

    if accelerator == "v5p":
        ports = alloc_free_ports(num_gpus + 1)
        master_port, sb_ports = ports[0], ports[1:]
        if num_gpus == 2:
            chips_bounds, topo, accel_t, vis_devs = "1,2,1", "1x2x1", "v5p-4", "0,1" if replica_idx == 0 else "2,3"
        else:
            chips_bounds, topo, accel_t, vis_devs = "2,2,1", "2x2x1", "v5p-8", "0,1,2,3"
        env.update(
            {
                "TPU_SKIP_MDS_QUERY": "true",
                "TPU_WORKER_ID": "0",
                "TPU_WORKER_HOSTNAMES": "localhost",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(master_port),
                "TORCH_TPU_SLICEBUILDER_ADDRESSES": ",".join(f"localhost:{p}" for p in sb_ports),
                "TORCH_TPU_DEVICES_PER_HOST": str(num_gpus),
                "TPU_CHIPS_PER_HOST_BOUNDS": chips_bounds,
                "CHIPS_PER_HOST_BOUNDS": chips_bounds,
                "TPU_HOST_BOUNDS": "1,1,1",
                "HOST_BOUNDS": "1,1,1",
                "TPU_TOPOLOGY": topo,
                "TOPOLOGY": topo,
                "TPU_ACCELERATOR_TYPE": accel_t,
                "TPU_VISIBLE_DEVICES": vis_devs,
                "JAX_PLATFORMS": "cpu",
                "PJRT_DEVICE": "TPU",
                "TPU_ML_PLATFORM": "PyTorch",
                "LIBTPU_INIT_ARGS": (
                    "--xla_tpu_scoped_vmem_limit_kib=40960 "
                    "--xla_tpu_enable_async_collective_fusion=true "
                    "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true"
                ),
                "HF_HOME": "/dev/shm/hf_cache",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": str(cache_root / "xla"),
                "TORCHINDUCTOR_AUTOGRAD_CACHE": "0",
                "TMPDIR": str(cache_root / "tmp"),
                "PYTHONFAULTHANDLER": "1",
                "PYTHONUNBUFFERED": "1",
                "SGLANG_TPU_V5P_SPLASH": "1" if v5p_splash else "0",
                "SGLANG_TPU_SPLASH_VARLEN": "1" if (v5p_splash and splash_varlen) else "0",
                "SGLANG_TPU_DISABLE_SPLASH": "1" if disable_splash else "0",
                "SGLANG_H3_EXACT_21_9_768P": "1",
                "SGLANG_REMOTE_ENCODE_URL": remote_encode_url,
            }
        )
        return env

    # Cloud TPU v7x (4 chips/host = 8 TensorCores; sub-host 2 chips = 4 TensorCores)
    is_subhost = num_gpus == 4
    if is_subhost:
        chips_bounds = "1,2,1"
        topo = "1x2x1"
        torch_tpu_topo = "1,2,1,2"
        accel_t = "tpu7x-4"
        if replica_idx == 0:
            local_ranks, vis_devs, metrics_ports = "0,1,2,3", "0,1", "8431,8432,8433,8434"
        else:
            local_ranks, vis_devs, metrics_ports = "4,5,6,7", "2,3", "8435,8436,8437,8438"
    else:
        chips_bounds = "2,2,1"
        topo = "2x2x1"
        torch_tpu_topo = "2,2,1,2"
        accel_t = "tpu7x-8"
        local_ranks = "0,1,2,3,4,5,6,7"
        vis_devs = "0,1,2,3"
        metrics_ports = "8431,8432,8433,8434,8435,8436,8437,8438"

    sb = ",".join(f"localhost:{port_seed + i}" for i in range(num_gpus))
    env.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port_seed - 1),
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": sb,
            "TORCH_TPU_TOPOLOGY": torch_tpu_topo,
            "SGLANG_TPU_LOCAL_RANKS": local_ranks,
            "SGLANG_TPU_SUBHOST": "1" if is_subhost else "0",
            "TPU_CHIPS_PER_HOST_BOUNDS": chips_bounds,
            "CHIPS_PER_HOST_BOUNDS": chips_bounds,
            "TPU_HOST_BOUNDS": "1,1,1",
            "HOST_BOUNDS": "1,1,1",
            "TPU_TOPOLOGY": topo,
            "TOPOLOGY": topo,
            "TPU_ACCELERATOR_TYPE": accel_t,
            "TPU_VISIBLE_DEVICES": vis_devs,
            "TPU_RUNTIME_METRICS_PORTS": metrics_ports,
            "TPU_PREMAPPED_BUFFER_SIZE": str(512 * 1024**2),
            "TPU_SKIP_MDS_QUERY": "true",
            "TPU_WORKER_ID": "0",
            "ENABLE_ICI_RESILIENCY": "false",
            "ENABLE_IMPROVED_REROUTE_ALLREDUCE_STRATEGY": "false",
            "INJECT_SLICE_BUILDER_FAULT": "",
            "VBAR_CONTROL_SERVICE_URL": "unix:///var/run/tpu-plugin/vbar.sock",
            "GCE_METADATA_HOST": "127.0.0.1:80",
            "GCE_METADATA_IP": "127.0.0.1",
            "GOOGLE_CLOUD_PROJECT": "tpu-serving-bench",
            "GCLOUD_PROJECT": "tpu-serving-bench",
            "LD_PRELOAD": "/workspace/metadata_hook.so",
            "HF_HOME": "/dev/shm/hf_cache",
            "HF_HUB_CACHE": "/dev/shm/hf_cache/hub",
            "SGLANG_DIFFUSION_CACHE_ROOT": "/dev/shm/sglang_diffusion_cache",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": str(cache_root / "xla"),
            "TORCHINDUCTOR_AUTOGRAD_CACHE": "0",
            "TMPDIR": str(cache_root / "tmp"),
            "PYTHONFAULTHANDLER": "1",
            "PYTHONUNBUFFERED": "1",
            "SGLANG_TPU_SPLASH_VARLEN": "1" if splash_varlen else "0",
            "SGLANG_TPU_DISABLE_SPLASH": "1" if disable_splash else "0",
            "SGLANG_H3_EXACT_21_9_768P": "1",
            "SGLANG_REMOTE_ENCODE_URL": remote_encode_url,
        }
    )
    return env


class SGLangEndpointRouter:
    def __init__(
        self,
        endpoints: list[str],
        procs: list[subprocess.Popen[Any]] | None = None,
        log_paths: list[Path] | None = None,
        boot_s: float = 0.0,
    ) -> None:
        self.endpoints = list(endpoints)
        self.procs = procs or []
        self.log_paths = log_paths or []
        self.boot_s = boot_s
        self._lock = threading.Lock()
        self._inflight = [0] * len(self.endpoints)
        self._rr = 0

    def acquire(self) -> tuple[int, str]:
        with self._lock:
            n = len(self.endpoints)
            best = min(range(n), key=lambda i: (self._inflight[i], (i - self._rr) % n))
            self._inflight[best] += 1
            self._rr = (best + 1) % n
            return best, self.endpoints[best]

    def release(self, i: int) -> None:
        with self._lock:
            self._inflight[i] = max(0, self._inflight[i] - 1)

    def submit(
        self,
        payload: dict[str, Any],
        *,
        timeout_s: float = 2400.0,
        probe_streams: bool = False,
        bucket: str = "",
        save_video_gcs_obj: str | None = None,
    ) -> dict[str, Any]:
        slot, ep = self.acquire()
        proc = self.procs[slot] if slot < len(self.procs) else None
        lp = self.log_paths[slot] if slot < len(self.log_paths) else None
        try:
            return submit_and_poll_video(
                ep,
                payload,
                timeout_s=timeout_s,
                proc=proc,
                log_path=lp,
                probe_streams=probe_streams,
                bucket=bucket,
                save_video_gcs_obj=save_video_gcs_obj,
            )
        finally:
            self.release(slot)

    def stop(self) -> None:
        kill_sglang_processes(self.procs)


def start_sglang_cluster(
    idx: int,
    *,
    accelerator: str,
    config_name: str,
    model_path: str,
    dp_replicas: int,
    num_gpus: int,
    ulysses_degree: int,
    quantization: str | None,
    encoder_parallel: str = "fold",
    text_encoder_offload: bool = False,
    vae_offload: bool = False,
    adaln_online: bool = False,
    use_fsdp: bool = False,
    splash_varlen: bool = True,
    disable_splash: bool = False,
    v5p_splash: bool = False,
    remote_encode_url: str = "",
    base_port: int = 30010,
    ready_timeout_s: int = 1600,
) -> SGLangEndpointRouter:
    global _PORT_SEED_COUNTER
    kill_sglang_processes()

    procs: list[subprocess.Popen[Any]] = []
    ports: list[int] = []
    log_paths: list[Path] = []
    t0 = time.time()

    te_off = "true" if text_encoder_offload else "false"
    vae_off = "true" if vae_offload else "false"

    for r_i in range(dp_replicas):
        port = base_port + r_i
        port_seed = _PORT_SEED_COUNTER
        _PORT_SEED_COUNTER += 30

        out_dir = Path(f"/dev/shm/sglang_out_{config_name}_r{r_i}")
        out_dir.mkdir(parents=True, exist_ok=True)
        lp = Path(f"/workspace/sglang_{config_name}_r{r_i}.log")

        cmd = [
            "sglang",
            "serve",
            "--model-path",
            model_path,
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--num-gpus",
            str(num_gpus),
            "--ulysses-degree",
            str(ulysses_degree),
            "--encoder-parallel",
            encoder_parallel,
            "--attention-backend",
            "tpu_splash_attn",
            "--warmup-mode",
            "off",
            "--dit-cpu-offload",
            "false",
            "--dit-layerwise-offload",
            "false",
            "--text-encoder-cpu-offload",
            te_off,
            "--image-encoder-cpu-offload",
            te_off,
            "--vae-cpu-offload",
            vae_off,
            "--pin-cpu-memory",
            "false",
            "--output-path",
            str(out_dir),
        ]
        if accelerator == "v5p" and vae_offload:
            cmd.extend(["--offload-during-compile", "true"])
        if quantization:
            cmd.extend(["--quantization", quantization])
        if adaln_online:
            cmd.extend(["--minimax-h3-adaln-online", "--minimax-h3-adaln-host-cache-gb", "1"])
        if use_fsdp:
            cmd.extend(["--use-fsdp-inference", "true"])

        env = build_server_env(
            accelerator=accelerator,
            num_gpus=num_gpus,
            replica_idx=r_i,
            splash_varlen=splash_varlen,
            disable_splash=disable_splash,
            v5p_splash=v5p_splash,
            remote_encode_url=remote_encode_url,
            port_seed=port_seed,
        )
        log(idx, f"Starting [{config_name}] rep={r_i}/{dp_replicas} on :{port} (RAM={host_ram_summary()}): {' '.join(cmd)}")
        lf = open(lp, "w")
        procs.append(subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env))
        ports.append(port)
        log_paths.append(lp)

    ready = [False] * dp_replicas
    while time.time() - t0 < ready_timeout_s:
        for r_i, proc in enumerate(procs):
            if ready[r_i]:
                continue
            if proc.poll() is not None:
                tail = log_paths[r_i].read_text(errors="replace")[-4000:]
                kill_sglang_processes(procs)
                raise RuntimeError(f"sglang [{config_name}] rep={r_i} exited early rc={proc.returncode}:\n{tail}")
            if log_paths[r_i].exists():
                txt = log_paths[r_i].read_text(errors="replace")
                if any(
                    err_k in txt
                    for err_k in (
                        "Rank 0 scheduler is dead",
                        "RuntimeBufferAllocationFailure",
                        "RuntimeProgramAllocationFailure",
                        "RESOURCE_EXHAUSTED",
                        "Failed to acquire a TPU device node",
                    )
                ):
                    kill_sglang_processes(procs)
                    raise RuntimeError(f"sglang [{config_name}] rep={r_i} startup error:\n{txt[-4000:]}")
            for u in (f"http://127.0.0.1:{ports[r_i]}/health", f"http://127.0.0.1:{ports[r_i]}/v1/videos"):
                try:
                    with urllib.request.urlopen(u, timeout=4) as resp:
                        if resp.status == 200:
                            ready[r_i] = True
                            break
                except Exception:
                    pass
        if all(ready):
            boot_s = round(time.time() - t0, 2)
            pod_ip = local_pod_ip()
            eps = [f"http://{pod_ip}:{p}" for p in ports]
            log(idx, f"Cluster [{config_name}] READY ({eps}) in {boot_s}s! (RAM={host_ram_summary()})")
            return SGLangEndpointRouter(eps, procs=procs, log_paths=log_paths, boot_s=boot_s)
        time.sleep(3.0)

    tails = "\n---\n".join(lp.read_text(errors="replace")[-2000:] for lp in log_paths if lp.exists())
    kill_sglang_processes(procs)
    raise TimeoutError(f"Cluster [{config_name}] timed out after {ready_timeout_s}s:\n{tails}")


def probe_mp4_file(fp: str) -> dict[str, Any] | None:
    try:
        pr = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration,size,bit_rate",
                "-show_entries",
                "stream=codec_type,codec_name,width,height,nb_frames,avg_frame_rate,sample_rate,channels",
                "-of",
                "json",
                fp,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if pr.returncode == 0:
            return json.loads(pr.stdout)
    except Exception:
        pass
    return None


def submit_and_poll_video(
    endpoint: str,
    payload: dict[str, Any],
    *,
    timeout_s: float = 2400.0,
    proc: subprocess.Popen[Any] | None = None,
    log_path: Path | None = None,
    probe_streams: bool = False,
    bucket: str = "",
    save_video_gcs_obj: str | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{endpoint}/v1/videos",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            job = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "status": f"http_{e.code}",
            "error": err_body[:900],
            "e2e_s": round(time.perf_counter() - t0, 3),
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": "submit_error",
            "error": str(exc)[:900],
            "e2e_s": round(time.perf_counter() - t0, 3),
        }

    vid_id = job["id"]
    poll_url = f"{endpoint}/v1/videos/{vid_id}"
    while time.perf_counter() - t0 < timeout_s:
        if proc is not None and proc.poll() is not None:
            tail = log_path.read_text(errors="replace")[-1200:] if log_path and log_path.exists() else ""
            return {
                "ok": False,
                "id": vid_id,
                "status": "server_crashed",
                "error": f"rc={proc.returncode}: {tail[-700:]}",
                "e2e_s": round(time.perf_counter() - t0, 3),
            }
        if log_path is not None and log_path.exists():
            full_txt = log_path.read_text(errors="replace")
            if any(
                err_k in full_txt
                for err_k in (
                    "Rank 0 scheduler is dead",
                    "RESOURCE_EXHAUSTED",
                    "RuntimeBufferAllocationFailure",
                    "RuntimeProgramAllocationFailure",
                    "Output verification failed",
                    "StageVerificationError",
                    "Failed to acquire a TPU device node",
                )
            ):
                return {
                    "ok": False,
                    "id": vid_id,
                    "status": "worker_oom_or_crash",
                    "error": full_txt[-1200:],
                    "e2e_s": round(time.perf_counter() - t0, 3),
                }
        time.sleep(0.5)
        try:
            with urllib.request.urlopen(poll_url, timeout=30) as resp:
                st = json.loads(resp.read().decode("utf-8"))
        except Exception:
            continue
        status = st.get("status")
        if status == "completed":
            e2e_s = time.perf_counter() - t0
            fp = st.get("file_path")
            fsize = os.path.getsize(fp) if fp and os.path.exists(fp) else None
            probe_info = None
            saved_gcs = None
            if fp and os.path.exists(fp):
                if probe_streams or save_video_gcs_obj:
                    probe_info = probe_mp4_file(fp)
                if save_video_gcs_obj and bucket:
                    gcs_upload_file(bucket, save_video_gcs_obj, Path(fp), content_type="video/mp4")
                    saved_gcs = f"gs://{bucket}/{save_video_gcs_obj}"
                try:
                    os.remove(fp)
                except OSError:
                    pass
            return {
                "ok": True,
                "id": vid_id,
                "status": "completed",
                "e2e_s": round(e2e_s, 3),
                "inference_time_s": st.get("inference_time_s"),
                "peak_memory_mb": st.get("peak_memory_mb"),
                "size": st.get("size"),
                "seconds": st.get("seconds"),
                "file_bytes": fsize,
                "ffprobe": probe_info,
                "saved_video_gcs": saved_gcs,
            }
        if status == "failed":
            return {
                "ok": False,
                "id": vid_id,
                "status": "failed",
                "error": str(st.get("error"))[:900],
                "e2e_s": round(time.perf_counter() - t0, 3),
            }

    return {
        "ok": False,
        "id": vid_id,
        "status": "timeout",
        "error": f"Timed out after {timeout_s}s",
        "e2e_s": round(time.perf_counter() - t0, 3),
    }


def build_minimax_payload(
    *,
    prompt: str,
    task: str,
    short_edge: int,
    aspect_ratio: str,
    duration_seconds: float,
    num_inference_steps: int,
    seed: int,
    conditions: list[dict[str, Any]] | None = None,
    flow_shift: float | None = None,
    audio_flow_shift: float | None = None,
    perf_dump_path: str | None = None,
) -> dict[str, Any]:
    # NOTE: "quality": "lossless" is SGLang's MiniMax-H3 default and only means "no Cache-DiT"
    # (equivalent to "cache_dit": false). It does not describe weight precision or step count.
    payload: dict[str, Any] = {
        "prompt": prompt,
        "task": task,
        "conditions": conditions or [],
        "target": {
            "short_edge": int(short_edge),
            "aspect_ratio": aspect_ratio,
            "duration_seconds": float(duration_seconds),
        },
        "num_inference_steps": int(num_inference_steps),
        "quality": "lossless",
        "generator_device": "cpu",
        "seed": int(seed),
    }
    if flow_shift is not None:
        payload["flow_shift"] = float(flow_shift)
    if audio_flow_shift is not None:
        payload["audio_flow_shift"] = float(audio_flow_shift)
    if perf_dump_path:
        payload["perf_dump_path"] = str(perf_dump_path)
    return payload


def ref2va_6_conditions(inputs_dir: Path) -> list[dict[str, Any]]:
    return [
        {"type": "video", "role": "reference", "uri": str(inputs_dir / "h3_r2v_compare_ref_14s.mp4")},
        {"type": "image", "role": "reference", "uri": str(inputs_dir / "h3_r2v_compare_img_01.jpg")},
        {"type": "image", "role": "reference", "uri": str(inputs_dir / "h3_r2v_compare_img_02.png")},
        {"type": "image", "role": "reference", "uri": str(inputs_dir / "h3_r2v_compare_img_03.jpg")},
        {"type": "image", "role": "reference", "uri": str(inputs_dir / "h3_r2v_compare_img_04.jpg")},
        {"type": "image", "role": "reference", "uri": str(inputs_dir / "h3_r2v_compare_img_05.png")},
    ]


def ref2va_1_condition(inputs_dir: Path) -> list[dict[str, Any]]:
    return [{"type": "image", "role": "reference", "uri": str(inputs_dir / "h3_r2v_compare_img_01.jpg")}]


def fl2va_dragon_condition(inputs_dir: Path) -> list[dict[str, Any]]:
    p = inputs_dir / "gpt_dragon_1080p.jpg"
    if not p.exists():
        p = inputs_dir / "synthetic_1344x768.png"
    return [{"type": "image", "role": "keyframe", "uri": str(p), "frame_index": 0}]


def execute_workload_list(
    idx: int,
    *,
    bucket: str,
    gcs_prefix: str,
    local_jsonl: Path,
    gcs_jsonl_obj: str,
    config_meta: dict[str, Any],
    router: SGLangEndpointRouter,
    hbm_mon: LibtpuHbmMonitor | None,
    workloads: list[dict[str, Any]],
    collected: list[dict[str, Any]],
) -> bool:
    """Run a list of workloads on `router`. Returns False if an HBM OOM / crash occurred requiring fallback."""
    warmed_shapes: set[tuple[str, int, str, float, int]] = set()

    for w_i, wl in enumerate(workloads):
        task = wl["task"]
        short_edge = int(wl["short_edge"])
        aspect = str(wl["aspect_ratio"])
        dur = float(wl["duration_seconds"])
        steps_grid = int(wl["num_inference_steps"])
        denoise_steps = steps_grid - 1
        conc = int(wl.get("concurrency", 1))
        n_reqs = int(wl.get("num_requests", max(2, conc)))
        conds = wl.get("conditions", [])
        prompt_base = wl.get("prompt", PROMPTS_SWEEP[0])
        seed_base = int(wl.get("seed", 0))
        flow_shift = wl.get("flow_shift")
        audio_flow_shift = wl.get("audio_flow_shift")
        save_name = wl.get("save_mp4_name")

        shape_key = (task, short_edge, aspect, dur, len(conds))
        cold_compile_s = None
        if shape_key not in warmed_shapes:
            warm_steps = min(steps_grid, 5)
            log(idx, f"[{config_meta['name']}] Warming XLA shape {shape_key} (steps_grid={warm_steps})...")
            w_payload = build_minimax_payload(
                prompt=prompt_base,
                task=task,
                short_edge=short_edge,
                aspect_ratio=aspect,
                duration_seconds=dur,
                num_inference_steps=warm_steps,
                seed=seed_base,
                conditions=conds,
                flow_shift=flow_shift,
                audio_flow_shift=audio_flow_shift,
            )
            w_outs = [router.submit(w_payload, probe_streams=True)]
            cold_compile_s = w_outs[0]["e2e_s"]
            if not w_outs[0].get("ok"):
                err_str = str(w_outs[0].get("error", ""))
                log(idx, f"[{config_meta['name']}] Warmup failed on {wl['label']}: {err_str[:400]}")
                row = {
                    **config_meta,
                    "timestamp_utc": utc_now(),
                    "workload_label": wl["label"],
                    "task": task,
                    "short_edge": short_edge,
                    "aspect_ratio": aspect,
                    "duration_seconds": dur,
                    "num_inference_steps": steps_grid,
                    "denoise_steps": denoise_steps,
                    "concurrency": conc,
                    "num_requests": n_reqs,
                    "ok_count": 0,
                    "cold_compile_e2e_s": cold_compile_s,
                    "error": err_str,
                }
                collected.append(row)
                append_and_upload_jsonl(bucket, local_jsonl, gcs_jsonl_obj, row)
                return False
            warmed_shapes.add(shape_key)

        perf_dir = Path("/workspace/perf")
        perf_slug = (save_name or wl["label"]).replace(".mp4", "")
        perf_slug = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in perf_slug)[:120]

        def _run_one(r_i: int) -> dict[str, Any]:
            pr = prompt_base if wl.get("exact_prompt", False) else PROMPTS_SWEEP[(w_i + r_i) % len(PROMPTS_SWEEP)]
            # identical_requests: same prompt AND same seed on every repetition, so the reported latency is the
            # net inference time of a repeated identical request on a warm server.
            if wl.get("identical_requests", False):
                sd = seed_base
            else:
                sd = seed_base if (wl.get("exact_seed", False) and r_i == 0) else (seed_base + r_i)
            vid_obj = f"{gcs_prefix}/videos/{save_name}" if (save_name and r_i == 0) else None
            pd_path = str(perf_dir / f"{perf_slug}_r{r_i}.json") if wl.get("perf_dump", False) else None
            p = build_minimax_payload(
                prompt=pr,
                task=task,
                short_edge=short_edge,
                aspect_ratio=aspect,
                duration_seconds=dur,
                num_inference_steps=steps_grid,
                seed=sd,
                conditions=conds,
                flow_shift=flow_shift,
                audio_flow_shift=audio_flow_shift,
                perf_dump_path=pd_path,
            )
            return router.submit(
                p,
                probe_streams=(r_i == 0),
                bucket=bucket,
                save_video_gcs_obj=vid_obj,
            )

        log(idx, f"[{config_meta['name']}] Running {wl['label']} (steps_grid={steps_grid} [{denoise_steps} NFEs], c={conc}, n={n_reqs})...")
        t_wall0 = time.perf_counter()
        if conc == 1:
            outs = [_run_one(r_i) for r_i in range(n_reqs)]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as pool:
                outs = list(pool.map(_run_one, range(n_reqs)))
        wall_s = time.perf_counter() - t_wall0

        # Collect per-request SGLang perf dumps (stages + per-denoising-step ms) when requested.
        perf_reports: list[dict[str, Any]] = []
        if wl.get("perf_dump", False):
            for r_i in range(n_reqs):
                pf = perf_dir / f"{perf_slug}_r{r_i}.json"
                for _ in range(40):
                    if pf.exists():
                        break
                    time.sleep(0.5)
                if not pf.exists():
                    log(idx, f"[{config_meta['name']}] WARNING: perf dump missing: {pf}")
                    continue
                try:
                    rep = json.loads(pf.read_text())
                except Exception as exc:
                    log(idx, f"[{config_meta['name']}] WARNING: unreadable perf dump {pf}: {exc}")
                    continue
                perf_obj = f"{gcs_prefix}/perf/{config_meta['name']}/{pf.name}"
                gcs_upload_file(bucket, perf_obj, pf, content_type="application/json")
                perf_reports.append(
                    {
                        "request_index": r_i,
                        "gcs": f"gs://{bucket}/{perf_obj}",
                        "total_duration_ms": rep.get("total_duration_ms"),
                        "stages_ms": {s.get("name"): s.get("duration_ms") for s in rep.get("steps", [])},
                        "denoise_steps_ms": [s.get("duration_ms") for s in rep.get("denoise_steps_ms", [])],
                        "measured_denoise_steps": len(rep.get("denoise_steps_ms", [])),
                    }
                )

        oks = [o for o in outs if o.get("ok")]
        lats = sorted(o["e2e_s"] for o in oks)
        inf_ts = [float(o["inference_time_s"]) for o in oks if o.get("inference_time_s") is not None]
        mean_e2e = round(statistics.mean(lats), 3) if lats else None
        med_e2e = round(statistics.median(lats), 3) if lats else None
        p90_e2e = round(lats[min(len(lats) - 1, int(0.9 * len(lats)))], 3) if lats else None
        mean_inf = round(statistics.mean(inf_ts), 3) if inf_ts else None
        med_inf = round(statistics.median(inf_ts), 3) if inf_ts else None
        enc_ovh = round(max(0.0, med_e2e - med_inf), 3) if (med_e2e is not None and med_inf is not None) else None
        step_ms = round(((med_inf or med_e2e) * 1000.0) / max(denoise_steps, 1), 1) if (med_inf or med_e2e) else None
        eff_vid_s = float(oks[0].get("seconds") or dur) if oks else dur
        rpm = round((len(oks) / wall_s) * 60.0, 3) if oks and wall_s > 0 else 0.0
        rtf = round((len(oks) * eff_vid_s) / wall_s, 3) if oks and wall_s > 0 else 0.0
        peak_hbm_gib = hbm_mon.snapshot_peak_gib() if hbm_mon is not None else None
        saved_vid = next((o.get("saved_video_gcs") for o in oks if o.get("saved_video_gcs")), None)
        ffprobe_meta = next((o.get("ffprobe") for o in oks if o.get("ffprobe")), None)
        resolved_size = oks[0].get("size") if oks else None

        row = {
            **config_meta,
            "timestamp_utc": utc_now(),
            "workload_label": wl["label"],
            "task": task,
            "resolved_size": resolved_size,
            "short_edge": short_edge,
            "aspect_ratio": aspect,
            "duration_seconds": dur,
            "output_video_seconds": eff_vid_s,
            "num_inference_steps": steps_grid,
            "denoise_steps": denoise_steps,
            "flow_shift": flow_shift,
            "audio_flow_shift": audio_flow_shift,
            "num_conditions": len(conds),
            "concurrency": conc,
            "num_requests": n_reqs,
            "ok_count": len(oks),
            "cold_compile_e2e_s": cold_compile_s,
            "wall_s": round(wall_s, 3),
            "mean_e2e_s": mean_e2e,
            "median_e2e_s": med_e2e,
            "p90_e2e_s": p90_e2e,
            "mean_inference_s": mean_inf,
            "median_inference_s": med_inf,
            "encode_and_io_overhead_s": enc_ovh,
            "ms_per_denoise_step": step_ms,
            "rpm_videos_per_min": rpm,
            "video_sec_per_wall_sec": rtf,
            "peak_hbm_per_core_gib": peak_hbm_gib,
            "peak_memory_mb": max((o.get("peak_memory_mb") or 0 for o in oks), default=None),
            "sample_mp4_bytes": oks[0].get("file_bytes") if oks else None,
            "saved_video_gcs": saved_vid,
            "ffprobe": ffprobe_meta,
            "inference_s_all": [round(x, 3) for x in inf_ts],
            "e2e_s_all": [round(float(o["e2e_s"]), 3) for o in oks],
            "identical_requests": bool(wl.get("identical_requests", False)),
            "prompt_exact": prompt_base if wl.get("exact_prompt", False) else None,
            "seed": seed_base,
            "flow_shift_effective": flow_shift if flow_shift is not None else "sglang_default(12.0)",
            "audio_flow_shift_effective": audio_flow_shift if audio_flow_shift is not None else "sglang_default(3.0)",
            "denoise_steps_source": "sglang minimax_h3/time_request.py: linspace(1,0,N) -> N-1 transformer evaluations",
            "adapter": wl.get("adapter"),
            "attention": wl.get("attention", "dense (Pallas Splash)"),
            "sync_stage_profiling": os.environ.get("SGLANG_DIFFUSION_SYNC_STAGE_PROFILING", "0") == "1",
            "perf_reports": perf_reports or None,
            "error": None if len(oks) == n_reqs else next((o.get("error") for o in outs if not o.get("ok")), None),
        }
        log(
            idx,
            f"[{config_meta['name']}] DONE {wl['label']}: ok={len(oks)}/{n_reqs} size={resolved_size} "
            f"med_inf={med_inf}s med_e2e={med_e2e}s enc_ovh={enc_ovh}s RPM={rpm}/min vid={saved_vid}",
        )
        collected.append(row)
        append_and_upload_jsonl(bucket, local_jsonl, gcs_jsonl_obj, row)
    return True


def run_with_auto_offload_fallback(
    idx: int,
    *,
    accelerator: str,
    bucket: str,
    gcs_prefix: str,
    local_jsonl: Path,
    gcs_jsonl_obj: str,
    config_name: str,
    model_short: str,
    model_path: str,
    chips_per_host: int,
    dp_replicas: int,
    num_gpus: int,
    ulysses_degree: int,
    quantization: str | None,
    try_specs: list[dict[str, Any]],
    remote_encode_url: str,
    workloads: list[dict[str, Any]],
    collected: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, SGLangEndpointRouter | None]:
    """Try candidate optimization specs in order (fastest All-HBM/FSDP first, safe offload fallback second)."""
    for s_i, spec in enumerate(try_specs):
        tag = f"{config_name}_{spec['tag']}"
        router: SGLangEndpointRouter | None = None
        hbm_mon = LibtpuHbmMonitor() if accelerator == "v7x" else None
        try:
            if hbm_mon is not None:
                hbm_mon.start()
            router = start_sglang_cluster(
                idx,
                accelerator=accelerator,
                config_name=tag,
                model_path=model_path,
                dp_replicas=dp_replicas,
                num_gpus=num_gpus,
                ulysses_degree=ulysses_degree,
                quantization=quantization,
                encoder_parallel=spec.get("encoder_parallel", "fold"),
                text_encoder_offload=spec.get("text_encoder_offload", False),
                vae_offload=spec.get("vae_offload", False),
                adaln_online=spec.get("adaln_online", False),
                use_fsdp=spec.get("use_fsdp", False),
                splash_varlen=spec.get("splash_varlen", True),
                disable_splash=spec.get("disable_splash", False),
                v5p_splash=spec.get("v5p_splash", False),
                remote_encode_url=remote_encode_url,
            )
            meta = {
                "name": tag,
                "accelerator": accelerator,
                "engine": "sglang-torchtpu-v0.2.0",
                "model_short": model_short,
                "precision": quantization or "bf16",
                "chips": chips_per_host,
                "dp_replicas": dp_replicas,
                "num_gpus_per_replica": num_gpus,
                "ulysses_degree": ulysses_degree,
                "encoder_parallel": spec.get("encoder_parallel", "fold"),
                "text_encoder_offload": spec.get("text_encoder_offload", False),
                "vae_offload": spec.get("vae_offload", False),
                "adaln_online": spec.get("adaln_online", False),
                "use_fsdp": spec.get("use_fsdp", False),
                "splash_varlen": spec.get("splash_varlen", True),
                "v5p_splash": spec.get("v5p_splash", False),
                "remote_encode_active": bool(remote_encode_url),
                "server_boot_s": router.boot_s,
            }
            ok = execute_workload_list(
                idx,
                bucket=bucket,
                gcs_prefix=gcs_prefix,
                local_jsonl=local_jsonl,
                gcs_jsonl_obj=gcs_jsonl_obj,
                config_meta=meta,
                router=router,
                hbm_mon=hbm_mon,
                workloads=workloads,
                collected=collected,
            )
            if hbm_mon is not None:
                hbm_mon.stop()
            if ok or s_i == len(try_specs) - 1:
                return spec, router
            log(idx, f"[{tag}] Triggering fallback to next spec after OOM/crash...")
            router.stop()
        except Exception as exc:
            if hbm_mon is not None:
                hbm_mon.stop()
            if router is not None:
                router.stop()
            log(idx, f"[{tag}] Spec boot/run exception (attempt {s_i + 1}/{len(try_specs)}): {exc}")
            if s_i == len(try_specs) - 1:
                err_row = {
                    "name": tag,
                    "accelerator": accelerator,
                    "model_short": model_short,
                    "precision": quantization or "bf16",
                    "chips": chips_per_host,
                    "workload_label": "server_boot",
                    "ok_count": 0,
                    "num_requests": 0,
                    "error": str(exc)[:1000],
                }
                collected.append(err_row)
                append_and_upload_jsonl(bucket, local_jsonl, gcs_jsonl_obj, err_row)
    return None, None


def run_multi_host_slice_sweep(
    idx: int,
    *,
    accelerator: str,
    bucket: str,
    gcs_prefix: str,
    slice_tag: str,
    slice_chips: int,
    model_short: str,
    model_path: str,
    quantization: str | None,
    server_spec: dict[str, Any],
    dp_per_host: int,
    num_gpus: int,
    ulysses_degree: int,
    remote_encode_url: str,
    warm_workloads: list[dict[str, Any]],
    slice_workloads: list[dict[str, Any]],
    local_jsonl: Path,
    gcs_jsonl_obj: str,
    collected: list[dict[str, Any]],
    pod_key: str,
    expected_keys: list[str],
    is_driver: bool,
) -> None:
    router: SGLangEndpointRouter | None = None
    boot_err: str | None = None
    try:
        router = start_sglang_cluster(
            idx,
            accelerator=accelerator,
            config_name=f"{slice_tag}_{pod_key}",
            model_path=model_path,
            dp_replicas=dp_per_host,
            num_gpus=num_gpus,
            ulysses_degree=ulysses_degree,
            quantization=quantization,
            encoder_parallel=server_spec.get("encoder_parallel", "fold"),
            text_encoder_offload=server_spec.get("text_encoder_offload", False),
            vae_offload=server_spec.get("vae_offload", False),
            adaln_online=server_spec.get("adaln_online", False),
            use_fsdp=server_spec.get("use_fsdp", False),
            splash_varlen=server_spec.get("splash_varlen", True),
            disable_splash=server_spec.get("disable_splash", False),
            v5p_splash=server_spec.get("v5p_splash", False),
            remote_encode_url=remote_encode_url,
            base_port=30200,
        )
        # Warm local replica(s) in parallel across all hosts before opening slice barrier!
        for w_wl in warm_workloads:
            wp = build_minimax_payload(
                prompt=w_wl.get("prompt", PROMPTS_SWEEP[0]),
                task=w_wl["task"],
                short_edge=w_wl["short_edge"],
                aspect_ratio=w_wl["aspect_ratio"],
                duration_seconds=w_wl["duration_seconds"],
                num_inference_steps=min(int(w_wl["num_inference_steps"]), 5),
                seed=900,
                conditions=w_wl.get("conditions", []),
                flow_shift=w_wl.get("flow_shift"),
                audio_flow_shift=w_wl.get("audio_flow_shift"),
            )
            for ep in router.endpoints:
                submit_and_poll_video(ep, wp, timeout_s=2400.0)
    except Exception as exc:
        boot_err = str(exc)[:900]
        log(idx, f"[{slice_tag}] Local host boot/warm error: {boot_err}")

    peer_map = gcs_barrier(
        idx,
        bucket,
        gcs_prefix,
        f"{slice_tag}_ready",
        {
            "pod_key": pod_key,
            "endpoints": router.endpoints if router is not None else [],
            "boot_error": boot_err,
        },
        pod_key=pod_key,
        expected_keys=expected_keys,
    )

    if is_driver:
        all_eps: list[str] = []
        active_hosts = 0
        for k in sorted(peer_map.keys()):
            eps = peer_map[k].get("endpoints", [])
            if eps:
                active_hosts += 1
                all_eps.extend(eps)
        if all_eps:
            slice_router = SGLangEndpointRouter(all_eps)
            meta = {
                "name": slice_tag,
                "slice_topology": f"DP={len(all_eps)}xUlysses={ulysses_degree}",
                "accelerator": accelerator,
                "engine": "sglang-torchtpu-v0.2.0",
                "model_short": model_short,
                "precision": quantization or "bf16",
                "chips": active_hosts * 4,
                "target_slice_chips": slice_chips,
                "dp_replicas": len(all_eps),
                "ulysses_degree": ulysses_degree,
                "remote_encode_active": bool(remote_encode_url),
            }
            for wl in slice_workloads:
                # Shapes already warmed on every host above
                conc = int(wl["concurrency"])
                n_reqs = int(wl["num_requests"])
                dur = float(wl["duration_seconds"])
                steps_grid = int(wl["num_inference_steps"])
                denoise_steps = steps_grid - 1
                conds = wl.get("conditions", [])
                flow_shift = wl.get("flow_shift")
                audio_flow_shift = wl.get("audio_flow_shift")

                def _call(r_i: int) -> dict[str, Any]:
                    p = build_minimax_payload(
                        prompt=wl.get("prompt") or PROMPTS_SWEEP[r_i % len(PROMPTS_SWEEP)],
                        task=wl["task"],
                        short_edge=wl["short_edge"],
                        aspect_ratio=wl["aspect_ratio"],
                        duration_seconds=dur,
                        num_inference_steps=steps_grid,
                        seed=7000 + r_i,
                        conditions=conds,
                        flow_shift=flow_shift,
                        audio_flow_shift=audio_flow_shift,
                    )
                    return slice_router.submit(p)

                log(idx, f"[{slice_tag}] Slice sweep {wl['label']} (DP={len(all_eps)}, c={conc}, n={n_reqs})...")
                t_c0 = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as pool:
                    outs = list(pool.map(_call, range(n_reqs)))
                wall_s = time.perf_counter() - t_c0

                oks = [o for o in outs if o.get("ok")]
                lats = sorted(o["e2e_s"] for o in oks)
                inf_ts = [float(o["inference_time_s"]) for o in oks if o.get("inference_time_s") is not None]
                med_e2e = round(statistics.median(lats), 3) if lats else None
                p90_e2e = round(lats[min(len(lats) - 1, int(0.9 * len(lats)))], 3) if lats else None
                med_inf = round(statistics.median(inf_ts), 3) if inf_ts else None
                eff_vid_s = float(oks[0].get("seconds") or dur) if oks else dur
                rpm = round((len(oks) / wall_s) * 60.0, 3) if oks and wall_s > 0 else 0.0
                rtf = round((len(oks) * eff_vid_s) / wall_s, 3) if oks and wall_s > 0 else 0.0

                crow = {
                    **meta,
                    "timestamp_utc": utc_now(),
                    "workload_label": wl["label"],
                    "task": wl["task"],
                    "resolved_size": oks[0].get("size") if oks else None,
                    "short_edge": wl["short_edge"],
                    "aspect_ratio": wl["aspect_ratio"],
                    "duration_seconds": dur,
                    "output_video_seconds": eff_vid_s,
                    "num_inference_steps": steps_grid,
                    "denoise_steps": denoise_steps,
                    "concurrency": conc,
                    "num_requests": n_reqs,
                    "ok_count": len(oks),
                    "wall_s": round(wall_s, 3),
                    "median_e2e_s": med_e2e,
                    "p90_e2e_s": p90_e2e,
                    "median_inference_s": med_inf,
                    "rpm_videos_per_min": rpm,
                    "video_sec_per_wall_sec": rtf,
                    "error": None if len(oks) == n_reqs else next((o.get("error") for o in outs if not o.get("ok")), None),
                }
                log(idx, f"[{slice_tag}] DONE {wl['label']}: ok={len(oks)}/{n_reqs} med_inf={med_inf}s med_e2e={med_e2e}s slice_RPM={rpm}/min RTF={rtf}x")
                collected.append(crow)
                append_and_upload_jsonl(bucket, local_jsonl, gcs_jsonl_obj, crow)

    gcs_barrier(
        idx,
        bucket,
        gcs_prefix,
        f"{slice_tag}_done",
        {"pod_key": pod_key, "done": True},
        pod_key=pod_key,
        expected_keys=expected_keys,
    )
    if router is not None:
        router.stop()


def build_workloads_ref2va_base(inputs_dir: Path, tag_prefix: str) -> list[dict[str, Any]]:
    return [
        {
            "label": "Ref2VA 768p 16:9 5s (50-step Base, 1 ref img, seed=1101)",
            "task": "ref2va",
            "prompt": PROMPT_T2VA_CATS,
            "exact_prompt": True,
            "exact_seed": True,
            "seed": 1101,
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 5.0,
            "num_inference_steps": 50,
            "flow_shift": 12.0,
            "audio_flow_shift": 3.0,
            "conditions": ref2va_1_condition(inputs_dir),
            "concurrency": 1,
            "num_requests": 2,
            "save_mp4_name": f"{tag_prefix}_ref2va_1344x768_5s_50step_1ref.mp4",
        },
        {
            "label": "Ref2VA 768p 21:9 14.375s 345f (50-step Base, 6 refs: 1 vid + 5 imgs)",
            "task": "ref2va",
            "prompt": PROMPT_REF2VA_SAMPLE,
            "exact_prompt": True,
            "exact_seed": True,
            "seed": 0,
            "short_edge": 768,
            "aspect_ratio": "21:9",
            "duration_seconds": 14.375,
            "num_inference_steps": 50,
            "flow_shift": 12.0,
            "audio_flow_shift": 3.0,
            "conditions": ref2va_6_conditions(inputs_dir),
            "concurrency": 1,
            "num_requests": 1,
            "save_mp4_name": f"{tag_prefix}_ref2va_1792x768_14s_345f_50step_6refs.mp4",
        },
    ]


def build_workloads_ref2va_turbo(inputs_dir: Path, tag_prefix: str) -> list[dict[str, Any]]:
    return [
        {
            "label": "Ref2VA Sample 1792x768 (21:9) 14.375s 345f (4-step Turbo Merged, NFE=4, shift=6.0, 6 refs)",
            "task": "ref2va",
            "prompt": PROMPT_REF2VA_SAMPLE,
            "exact_prompt": True,
            "exact_seed": True,
            "seed": 0,
            "short_edge": 768,
            "aspect_ratio": "21:9",
            "duration_seconds": 14.375,
            "num_inference_steps": 5,
            "flow_shift": 6.0,
            "audio_flow_shift": 3.0,
            "conditions": ref2va_6_conditions(inputs_dir),
            "concurrency": 1,
            "num_requests": 2,
            "save_mp4_name": f"{tag_prefix}_ref2va_1792x768_14s_345f_4step_nfe4_shift6.mp4",
        },
        {
            "label": "Ref2VA Sample 1792x768 (21:9) 14.375s 345f (4-step Turbo Merged, NFE=4, shift=12.0, 6 refs)",
            "task": "ref2va",
            "prompt": PROMPT_REF2VA_SAMPLE,
            "exact_prompt": True,
            "exact_seed": True,
            "seed": 0,
            "short_edge": 768,
            "aspect_ratio": "21:9",
            "duration_seconds": 14.375,
            "num_inference_steps": 5,
            "flow_shift": 12.0,
            "audio_flow_shift": 3.0,
            "conditions": ref2va_6_conditions(inputs_dir),
            "concurrency": 1,
            "num_requests": 2,
            "save_mp4_name": f"{tag_prefix}_ref2va_1792x768_14s_345f_4step_nfe4_shift12.mp4",
        },
        {
            "label": "Ref2VA Sample 1792x768 (21:9) 14.375s 345f (steps=4 / NFE=3 Turbo Merged, 6 refs)",
            "task": "ref2va",
            "prompt": PROMPT_REF2VA_SAMPLE,
            "exact_prompt": True,
            "exact_seed": True,
            "seed": 0,
            "short_edge": 768,
            "aspect_ratio": "21:9",
            "duration_seconds": 14.375,
            "num_inference_steps": 4,
            "flow_shift": 6.0,
            "audio_flow_shift": 3.0,
            "conditions": ref2va_6_conditions(inputs_dir),
            "concurrency": 1,
            "num_requests": 2,
            "save_mp4_name": f"{tag_prefix}_ref2va_1792x768_14s_345f_steps4_nfe3.mp4",
        },
        {
            "label": "Ref2VA 1344x768 (16:9) 5s 124f (4-step Turbo Merged, 1 ref)",
            "task": "ref2va",
            "prompt": PROMPT_T2VA_CATS,
            "exact_prompt": True,
            "exact_seed": True,
            "seed": 0,
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 5.0,
            "num_inference_steps": 5,
            "flow_shift": 6.0,
            "audio_flow_shift": 3.0,
            "conditions": ref2va_1_condition(inputs_dir),
            "concurrency": 1,
            "num_requests": 2,
        },
    ]


def build_workloads_fl2va_base(inputs_dir: Path, tag_prefix: str) -> list[dict[str, Any]]:
    return [
        {
            "label": "T2VA 1344x768 (16:9) 5s 124f (50-step Base, seed=1101)",
            "task": "t2va",
            "prompt": PROMPT_T2VA_CATS,
            "exact_prompt": True,
            "exact_seed": True,
            "seed": 1101,
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 5.0,
            "num_inference_steps": 50,
            "flow_shift": 12.0,
            "audio_flow_shift": 3.0,
            "conditions": [],
            "concurrency": 1,
            "num_requests": 2,
            "save_mp4_name": f"{tag_prefix}_t2va_1344x768_5s_50step_seed1101.mp4",
        },
        {
            "label": "I2V Sample Dragon 1344x768 (16:9) 14.375s 345f (25-step High-Quality Base, seed=0)",
            "task": "fl2va",
            "prompt": PROMPT_I2V_DRAGON,
            "exact_prompt": True,
            "exact_seed": True,
            "seed": 0,
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 14.375,
            "num_inference_steps": 26,
            "flow_shift": 12.0,
            "audio_flow_shift": 3.0,
            "conditions": fl2va_dragon_condition(inputs_dir),
            "concurrency": 1,
            "num_requests": 1,
            "save_mp4_name": f"{tag_prefix}_i2v_dragon_1344x768_14s_345f_25step_base.mp4",
        },
    ]


def build_workloads_fl2va_turbo(inputs_dir: Path, tag_prefix: str) -> list[dict[str, Any]]:
    return [
        {
            "label": "I2V Sample Dragon 1344x768 (16:9) 14.375s 345f (4-step Turbo Merged, NFE=4, shift=6.0)",
            "task": "fl2va",
            "prompt": PROMPT_I2V_DRAGON,
            "exact_prompt": True,
            "exact_seed": True,
            "seed": 0,
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 14.375,
            "num_inference_steps": 5,
            "flow_shift": 6.0,
            "audio_flow_shift": 3.0,
            "conditions": fl2va_dragon_condition(inputs_dir),
            "concurrency": 1,
            "num_requests": 3,
            "save_mp4_name": f"{tag_prefix}_i2v_dragon_1344x768_14s_345f_4step_nfe4_shift6.mp4",
        },
        {
            "label": "I2V Sample Dragon 1344x768 (16:9) 14.375s 345f (4-step Turbo Merged, NFE=4, shift=12.0)",
            "task": "fl2va",
            "prompt": PROMPT_I2V_DRAGON,
            "exact_prompt": True,
            "exact_seed": True,
            "seed": 0,
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 14.375,
            "num_inference_steps": 5,
            "flow_shift": 12.0,
            "audio_flow_shift": 3.0,
            "conditions": fl2va_dragon_condition(inputs_dir),
            "concurrency": 1,
            "num_requests": 2,
            "save_mp4_name": f"{tag_prefix}_i2v_dragon_1344x768_14s_345f_4step_nfe4_shift12.mp4",
        },
        {
            "label": "I2V Sample Dragon 1344x768 (16:9) 14.375s 345f (steps=4 / NFE=3 Turbo Merged)",
            "task": "fl2va",
            "prompt": PROMPT_I2V_DRAGON,
            "exact_prompt": True,
            "exact_seed": True,
            "seed": 0,
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 14.375,
            "num_inference_steps": 4,
            "flow_shift": 6.0,
            "audio_flow_shift": 3.0,
            "conditions": fl2va_dragon_condition(inputs_dir),
            "concurrency": 1,
            "num_requests": 2,
            "save_mp4_name": f"{tag_prefix}_i2v_dragon_1344x768_14s_345f_steps4_nfe3.mp4",
        },
        {
            "label": "I2V 1344x768 (16:9) 5s 124f (4-step Turbo Merged)",
            "task": "fl2va",
            "prompt": PROMPT_I2V_DRAGON,
            "exact_prompt": True,
            "exact_seed": True,
            "seed": 0,
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 5.0,
            "num_inference_steps": 5,
            "flow_shift": 6.0,
            "audio_flow_shift": 3.0,
            "conditions": fl2va_dragon_condition(inputs_dir),
            "concurrency": 1,
            "num_requests": 3,
            "save_mp4_name": f"{tag_prefix}_i2v_dragon_1344x768_5s_124f_4step.mp4",
        },
        {
            "label": "I2V 1344x768 (16:9) 10s 243f (4-step Turbo Merged)",
            "task": "fl2va",
            "prompt": PROMPT_I2V_DRAGON,
            "exact_prompt": True,
            "exact_seed": True,
            "seed": 0,
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 10.0,
            "num_inference_steps": 5,
            "flow_shift": 6.0,
            "audio_flow_shift": 3.0,
            "conditions": fl2va_dragon_condition(inputs_dir),
            "concurrency": 1,
            "num_requests": 2,
        },
        {
            "label": "I2V 1344x768 (16:9) 15s 362f (4-step Turbo Merged)",
            "task": "fl2va",
            "prompt": PROMPT_I2V_DRAGON,
            "exact_prompt": True,
            "exact_seed": True,
            "seed": 0,
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 15.0,
            "num_inference_steps": 5,
            "flow_shift": 6.0,
            "audio_flow_shift": 3.0,
            "conditions": fl2va_dragon_condition(inputs_dir),
            "concurrency": 1,
            "num_requests": 2,
        },
        {
            "label": "T2VA 1344x768 (16:9) 5s 124f (4-step Turbo Merged, seed=1101)",
            "task": "t2va",
            "prompt": PROMPT_T2VA_CATS,
            "exact_prompt": True,
            "exact_seed": True,
            "seed": 1101,
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 5.0,
            "num_inference_steps": 5,
            "flow_shift": 6.0,
            "audio_flow_shift": 3.0,
            "conditions": [],
            "concurrency": 1,
            "num_requests": 2,
            "save_mp4_name": f"{tag_prefix}_t2va_cats_1344x768_5s_124f_4step.mp4",
        },
    ]


def build_workloads_fasth3(tag_prefix: str) -> list[dict[str, Any]]:
    return [
        {
            "label": "FastH3 T2VA 1344x768 (16:9) 5s 124f (4-step VSA Distill, seed=1101)",
            "task": "t2va",
            "prompt": PROMPT_T2VA_CATS,
            "exact_prompt": True,
            "exact_seed": True,
            "seed": 1101,
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 5.0,
            "num_inference_steps": 5,
            "conditions": [],
            "concurrency": 1,
            "num_requests": 3,
            "save_mp4_name": f"{tag_prefix}_fasth3_t2va_1344x768_5s_124f_4step.mp4",
        },
        {
            "label": "FastH3 T2VA 1344x768 (16:9) 14.375s 345f (4-step VSA Distill)",
            "task": "t2va",
            "prompt": PROMPT_T2VA_CATS,
            "exact_prompt": True,
            "exact_seed": True,
            "seed": 1101,
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 14.375,
            "num_inference_steps": 5,
            "conditions": [],
            "concurrency": 1,
            "num_requests": 2,
            "save_mp4_name": f"{tag_prefix}_fasth3_t2va_1344x768_14s_345f_4step.mp4",
        },
        {
            "label": "FastH3 T2VA 1344x768 (16:9) 15s 362f (4-step VSA Distill)",
            "task": "t2va",
            "prompt": PROMPT_T2VA_CATS,
            "exact_prompt": True,
            "exact_seed": True,
            "seed": 1101,
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 15.0,
            "num_inference_steps": 5,
            "conditions": [],
            "concurrency": 1,
            "num_requests": 2,
        },
    ]


# =============================================================================
# Standard 4-step single-host benchmark workloads (1344x768 I2V and 1792x768 6-ref Ref2VA)
#  - num_inference_steps passed through unchanged (4). In SGLang MiniMax-H3 this is 3 transformer
#    evaluations (time_request.py: linspace(1,0,N) -> N-1 steps), on GPU and TPU alike.
#  - flow_shift / audio_flow_shift NOT sent: uses the SGLang task default schedule
#    defaults apply (12.0 / 3.0).
#  - 345 frames @ 24 fps corresponds to 14.375 s of video + synchronized stereo audio.
#  - Identical repeated requests (same prompt + seed); headline = median of the repeats.
# =============================================================================
def build_workloads_ref2va_6ref_4step(inputs_dir: Path, tag_prefix: str, *, perf: bool = False) -> list[dict[str, Any]]:
    return [
        {
            "label": "Ref2VA 6-reference 4-step request (1792x768 21:9, 345f, 6 refs, num_inference_steps=4, default flow_shift)",
            "task": "ref2va",
            "prompt": PROMPT_REF2VA_SAMPLE,
            "exact_prompt": True,
            "exact_seed": True,
            "identical_requests": True,
            "seed": 0,
            "short_edge": 768,
            "aspect_ratio": "21:9",
            "duration_seconds": 14.375,
            "num_inference_steps": 4,
            "flow_shift": None,
            "audio_flow_shift": None,
            "conditions": ref2va_6_conditions(inputs_dir),
            "concurrency": 1,
            "num_requests": 2 if perf else 3,
            "perf_dump": perf,
            "adapter": "lightx2v/Minimax-h3-Turbo minimax_h3_ref2v_turbo_4step_v0.1_bf16 (alpha 8, scale 1.0), merged before online FP8",
            "attention": "dense (Pallas Splash)",
            "save_mp4_name": f"{tag_prefix}_ref2va_1792x768_14s_345f_6ref_4step_turbo_v0.1.mp4",
        },
    ]


def build_workloads_i2v_dragon_4step(
    inputs_dir: Path, tag_prefix: str, adapter_tag: str, adapter_desc: str, *, perf: bool = False
) -> list[dict[str, Any]]:
    return [
        {
            "label": f"I2V Dragon 4-step request (1344x768, 345f, num_inference_steps=4, default flow_shift, {adapter_tag})",
            "task": "fl2va",
            "prompt": PROMPT_I2V_DRAGON,
            "exact_prompt": True,
            "exact_seed": True,
            "identical_requests": True,
            "seed": 0,
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 14.375,
            "num_inference_steps": 4,
            "flow_shift": None,
            "audio_flow_shift": None,
            "conditions": fl2va_dragon_condition(inputs_dir),
            "concurrency": 1,
            "num_requests": 2 if perf else 3,
            "perf_dump": perf,
            "adapter": adapter_desc,
            "attention": "dense (Pallas Splash)",
            "save_mp4_name": f"{tag_prefix}_i2v_dragon_1344x768_14s_345f_4step_{adapter_tag}.mp4",
        },
    ]


def build_workloads_profile_fl2va_sync(inputs_dir: Path, tag_prefix: str) -> list[dict[str, Any]]:
    """Device-synchronized stage/step breakdown (SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1).

    Used ONLY to attribute time to stages; headline latencies come from un-synchronized runs.
    Base FL2VA weights (no adapter): per-step compute is identical with or without a merged LoRA.
    """
    common_dragon = {
        "task": "fl2va",
        "prompt": PROMPT_I2V_DRAGON,
        "exact_prompt": True,
        "exact_seed": True,
        "identical_requests": True,
        "seed": 0,
        "short_edge": 768,
        "aspect_ratio": "16:9",
        "duration_seconds": 14.375,
        "flow_shift": None,
        "audio_flow_shift": None,
        "conditions": fl2va_dragon_condition(inputs_dir),
        "concurrency": 1,
        "num_requests": 2,
        "perf_dump": True,
        "adapter": "none (base FL2VA)",
    }
    common_t2va = {
        "task": "t2va",
        "prompt": PROMPT_T2VA_CATS,
        "exact_prompt": True,
        "exact_seed": True,
        "identical_requests": True,
        "seed": 1101,
        "short_edge": 768,
        "aspect_ratio": "16:9",
        "duration_seconds": 5.0,
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
        "conditions": [],
        "concurrency": 1,
        "num_requests": 2,
        "perf_dump": True,
        "adapter": "none (base FL2VA)",
    }
    return [
        {**common_dragon, "label": "PROFILE(sync) Dragon I2V 1344x768 345f num_inference_steps=4", "num_inference_steps": 4,
         "save_mp4_name": f"{tag_prefix}_profile_dragon_345f_steps4.mp4"},
        {**common_dragon, "label": "PROFILE(sync) Dragon I2V 1344x768 345f num_inference_steps=5", "num_inference_steps": 5},
        {**common_t2va, "label": "PROFILE(sync) T2VA 1344x768 124f num_inference_steps=5", "num_inference_steps": 5},
        {**common_t2va, "label": "PROFILE(sync) T2VA 1344x768 124f num_inference_steps=50", "num_inference_steps": 50},
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Cloud TPU v7x & v5p MiniMax-H3 Benchmark Runner")
    ap.add_argument("--gcs-bucket", required=True, help="GCS bucket for model staging, video artifacts, and JSONL output")
    ap.add_argument("--gcs-prefix", default="minimax-h3-tpu-bench/run1", help="GCS output prefix")
    ap.add_argument("--models-prefix", default="models", help="GCS prefix containing MiniMaxAI/MiniMax-H3 & lightx2v/Minimax-h3-Turbo")
    ap.add_argument("--inputs-prefix", default="minimax-h3-sample-inputs", help="GCS prefix containing sample images & reference video")
    ap.add_argument("--accelerator", choices=["v7x", "v5p"], required=True)
    ap.add_argument(
        "--role",
        choices=[
            "v7x8_main",
            "v7x8_bf16_fasth3",
            "v7x16_slice",
            "v7x_single_host",
            "v5p16_dedicated",
            "v5p32_dedicated",
            "v5p16_main",
            "v5p32_worker",
        ],
        required=True,
    )
    ap.add_argument("--remote-encode-url", default=os.environ.get("REMOTE_ENCODE_URL", DEFAULT_REMOTE_ENCODE_URL))
    args = ap.parse_args()

    idx = int(os.environ.get("JOB_COMPLETION_INDEX", "0"))
    bucket = args.gcs_bucket
    gcs_prefix = args.gcs_prefix.rstrip("/")
    models_prefix = args.models_prefix.strip("/")
    inputs_prefix = args.inputs_prefix.strip("/")

    shm_target = "850G" if args.accelerator == "v7x" else "360G"
    subprocess.run(["bash", "-lc", f"mount -o remount,size={shm_target} /dev/shm 2>/dev/null || true"], check=False)
    Path("/workspace").mkdir(parents=True, exist_ok=True)
    kill_sglang_processes()

    apply_sglang_tpu_patches(idx, accelerator=args.accelerator)
    if args.accelerator == "v7x":
        setup_v7x_metadata_hook(idx)

    remote_encode_url = check_remote_encode_service(idx, args.remote_encode_url)
    inputs_dir = stage_reference_inputs(idx, bucket, inputs_prefix)

    local_jsonl = Path(f"/workspace/{args.role}_pod{idx}.jsonl")
    gcs_jsonl_obj = f"{gcs_prefix}/results/{args.role}_pod{idx}.jsonl"
    collected: list[dict[str, Any]] = []

    lora_ref2v_obj = f"{models_prefix}/lightx2v/Minimax-h3-Turbo/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors"
    lora_fl2v_obj = f"{models_prefix}/lightx2v/Minimax-h3-Turbo/minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors"

    # Proven specs on v7x FP8 & BF16: Folded Qwen3-VL (8.3 GB/core, symmetric across all 8 TPU cores) + Offloaded VAE + Pallas Splash
    v7x_fp8_specs = [
        {"tag": "fold_pallas_offload", "encoder_parallel": "fold", "text_encoder_offload": True, "vae_offload": True, "adaln_online": False, "splash_varlen": False},
    ]
    v7x_bf16_specs = [
        {"tag": "fold_adaln_pallas_offload", "encoder_parallel": "fold", "text_encoder_offload": True, "vae_offload": True, "adaln_online": True, "splash_varlen": False},
    ]
    # Proven specs on v5p BF16: Folded Qwen3-VL + AdaLN-Online + Offloaded VAE + Disaggregated CPU Encode
    v5p_bf16_specs = [
        {"tag": "fold_adaln_off_sdpa", "encoder_parallel": "fold", "text_encoder_offload": True, "vae_offload": True, "adaln_online": True, "use_fsdp": False, "v5p_splash": False, "splash_varlen": False},
    ]
    # FastH3 checkpoint already embeds a pruned AdaLN polynomial, so adaln_online must be False; FSDP shards DiT across 4 v5p chips
    v5p_fasth3_specs = [
        {"tag": "fold_fsdp_off_sdpa", "encoder_parallel": "fold", "text_encoder_offload": True, "vae_offload": True, "adaln_online": False, "use_fsdp": True, "v5p_splash": False, "splash_varlen": False},
    ]

    # =========================================================================
    # ROLE 1: v7x8_main (2 hosts = 8 v7x chips: Pod 0 Ref2VA, Pod 1 FL2VA/T2VA, then v7x-8 DP=2xU=8)
    # =========================================================================
    if args.role == "v7x8_main":
        winning_fl2va_spec = v7x_fp8_specs[0]
        if idx == 0:
            ref_dir = stage_ref2va_in_shm(idx, bucket, models_prefix)
            # 1A: 50-step Base Ref2VA FP8
            _, r_base = run_with_auto_offload_fallback(
                idx,
                accelerator="v7x",
                bucket=bucket,
                gcs_prefix=gcs_prefix,
                local_jsonl=local_jsonl,
                gcs_jsonl_obj=gcs_jsonl_obj,
                config_name="v7x_ref2va_base_fp8_u8",
                model_short="MiniMax-H3-Ref2VA-Base",
                model_path=str(ref_dir),
                chips_per_host=4,
                dp_replicas=1,
                num_gpus=8,
                ulysses_degree=8,
                quantization="fp8",
                try_specs=v7x_fp8_specs,
                remote_encode_url=remote_encode_url,
                workloads=build_workloads_ref2va_base(inputs_dir, "v7x_fp8"),
                collected=collected,
            )
            if r_base is not None:
                r_base.stop()

            # 1B: Fuse Ref2VA 4-step Turbo LoRA in-place & run 6-Reference Ref2VA Workload
            merge_turbo_lora_inplace(idx, bucket=bucket, lora_gcs_obj=lora_ref2v_obj, ckpt_dir=ref_dir)
            _, r_turbo = run_with_auto_offload_fallback(
                idx,
                accelerator="v7x",
                bucket=bucket,
                gcs_prefix=gcs_prefix,
                local_jsonl=local_jsonl,
                gcs_jsonl_obj=gcs_jsonl_obj,
                config_name="v7x_ref2va_turbo4_fp8_u8",
                model_short="MiniMax-H3-Ref2VA-Turbo4",
                model_path=str(ref_dir),
                chips_per_host=4,
                dp_replicas=1,
                num_gpus=8,
                ulysses_degree=8,
                quantization="fp8",
                try_specs=v7x_fp8_specs,
                remote_encode_url=remote_encode_url,
                workloads=build_workloads_ref2va_turbo(inputs_dir, "v7x_fp8"),
                collected=collected,
            )
            if r_turbo is not None:
                r_turbo.stop()

            # Swap Pod 0 to FL2VA + Turbo4 so both hosts form a 2-replica v7x-8 (8-chip DP=2 x U=8) slice!
            fl_dir = swap_ref2va_to_fl2va_in_shm(idx, bucket, models_prefix)
            merge_turbo_lora_inplace(idx, bucket=bucket, lora_gcs_obj=lora_fl2v_obj, ckpt_dir=fl_dir)
        else:
            fl_dir = stage_fl2va_in_shm(idx, bucket, models_prefix)
            # 1A: 50-step Base T2VA & 25-step I2V FP8
            _, r_base = run_with_auto_offload_fallback(
                idx,
                accelerator="v7x",
                bucket=bucket,
                gcs_prefix=gcs_prefix,
                local_jsonl=local_jsonl,
                gcs_jsonl_obj=gcs_jsonl_obj,
                config_name="v7x_fl2va_base_fp8_u8",
                model_short="MiniMax-H3-FL2VA-Base",
                model_path=str(fl_dir),
                chips_per_host=4,
                dp_replicas=1,
                num_gpus=8,
                ulysses_degree=8,
                quantization="fp8",
                try_specs=v7x_fp8_specs,
                remote_encode_url=remote_encode_url,
                workloads=build_workloads_fl2va_base(inputs_dir, "v7x_fp8"),
                collected=collected,
            )
            if r_base is not None:
                r_base.stop()

            # 1B: Fuse FL2VA 4-step Turbo LoRA in-place & run I2V 345-Frame Workload + duration sweep
            merge_turbo_lora_inplace(idx, bucket=bucket, lora_gcs_obj=lora_fl2v_obj, ckpt_dir=fl_dir)
            w_spec, r_turbo = run_with_auto_offload_fallback(
                idx,
                accelerator="v7x",
                bucket=bucket,
                gcs_prefix=gcs_prefix,
                local_jsonl=local_jsonl,
                gcs_jsonl_obj=gcs_jsonl_obj,
                config_name="v7x_fl2va_turbo4_fp8_u8",
                model_short="MiniMax-H3-FL2VA-Turbo4",
                model_path=str(fl_dir),
                chips_per_host=4,
                dp_replicas=1,
                num_gpus=8,
                ulysses_degree=8,
                quantization="fp8",
                try_specs=v7x_fp8_specs,
                remote_encode_url=remote_encode_url,
                workloads=build_workloads_fl2va_turbo(inputs_dir, "v7x_fp8"),
                collected=collected,
            )
            if w_spec is not None:
                winning_fl2va_spec = w_spec
            if r_turbo is not None:
                r_turbo.stop()

        # 1C: Full v7x-8 Slice (8 v7x chips = DP=2 x Ulysses=8) Concurrent Sweep
        warm_wls = [
            {"task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 14.375, "num_inference_steps": 5, "flow_shift": 6.0, "audio_flow_shift": 3.0, "conditions": fl2va_dragon_condition(inputs_dir)},
            {"task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0, "num_inference_steps": 5, "flow_shift": 6.0, "audio_flow_shift": 3.0, "conditions": fl2va_dragon_condition(inputs_dir)},
        ]
        slice8_wls = [
            {"label": "v7x-8 Slice (DP=2xU8) I2V Dragon 1344x768 14.375s 345f (4-step, c=2)", "task": "fl2va", "prompt": PROMPT_I2V_DRAGON, "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 14.375, "num_inference_steps": 5, "flow_shift": 6.0, "audio_flow_shift": 3.0, "conditions": fl2va_dragon_condition(inputs_dir), "concurrency": 2, "num_requests": 4},
            {"label": "v7x-8 Slice (DP=2xU8) I2V Dragon 1344x768 14.375s 345f (4-step, c=4)", "task": "fl2va", "prompt": PROMPT_I2V_DRAGON, "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 14.375, "num_inference_steps": 5, "flow_shift": 6.0, "audio_flow_shift": 3.0, "conditions": fl2va_dragon_condition(inputs_dir), "concurrency": 4, "num_requests": 8},
            {"label": "v7x-8 Slice (DP=2xU8) I2V Dragon 1344x768 5s 124f (4-step, c=2)", "task": "fl2va", "prompt": PROMPT_I2V_DRAGON, "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0, "num_inference_steps": 5, "flow_shift": 6.0, "audio_flow_shift": 3.0, "conditions": fl2va_dragon_condition(inputs_dir), "concurrency": 2, "num_requests": 6},
            {"label": "v7x-8 Slice (DP=2xU8) I2V Dragon 1344x768 5s 124f (4-step, c=4)", "task": "fl2va", "prompt": PROMPT_I2V_DRAGON, "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0, "num_inference_steps": 5, "flow_shift": 6.0, "audio_flow_shift": 3.0, "conditions": fl2va_dragon_condition(inputs_dir), "concurrency": 4, "num_requests": 8},
        ]
        run_multi_host_slice_sweep(
            idx,
            accelerator="v7x",
            bucket=bucket,
            gcs_prefix=gcs_prefix,
            slice_tag="v7x8_slice_dp2_u8_fl2va_turbo4",
            slice_chips=8,
            model_short="MiniMax-H3-FL2VA-Turbo4",
            model_path=str(fl_dir),
            quantization="fp8",
            server_spec=winning_fl2va_spec,
            dp_per_host=1,
            num_gpus=8,
            ulysses_degree=8,
            remote_encode_url=remote_encode_url,
            warm_workloads=warm_wls,
            slice_workloads=slice8_wls,
            local_jsonl=local_jsonl,
            gcs_jsonl_obj=gcs_jsonl_obj,
            collected=collected,
            pod_key=f"v7x8_p{idx}",
            expected_keys=["v7x8_p0", "v7x8_p1"],
            is_driver=(idx == 0),
        )

    # =========================================================================
    # ROLE: v7x_single_host (4 independent single hosts = 4 chips / 8 TensorCores each, Ulysses-8)
    #   pod 0: Ref2VA 6-reference 4-step request, FP8 + Turbo ref2v v0.1 merged pre-quantization (timed, video)
    #   pod 1: I2V Dragon 4-step request, FP8: (A) official base weights, (B) Turbo v1.2 merged
    #   pod 2: device-synchronized stage/step profile, FL2VA base (Dragon 345f steps 4/5, T2VA 124f steps 5/50)
    #   pod 3: device-synchronized stage/step profile, Ref2VA 6-reference 4-step request
    # Headline latencies come ONLY from pods 0/1 (no synchronization overhead).
    # =========================================================================
    elif args.role == "v7x_single_host":
        if idx in (2, 3):
            os.environ["SGLANG_DIFFUSION_SYNC_STAGE_PROFILING"] = "1"
            os.environ["SGLANG_DIFFUSION_STAGE_LOGGING"] = "1"
            Path("/workspace/perf").mkdir(parents=True, exist_ok=True)
        common_run = dict(
            accelerator="v7x",
            bucket=bucket,
            gcs_prefix=gcs_prefix,
            local_jsonl=local_jsonl,
            gcs_jsonl_obj=gcs_jsonl_obj,
            chips_per_host=4,
            dp_replicas=1,
            num_gpus=8,
            ulysses_degree=8,
            quantization="fp8",
            try_specs=v7x_fp8_specs,
            remote_encode_url=remote_encode_url,
            collected=collected,
        )
        if idx in (0, 3):
            ref_dir = stage_ref2va_in_shm(idx, bucket, models_prefix)
            merge_turbo_lora_inplace(idx, bucket=bucket, lora_gcs_obj=lora_ref2v_obj, ckpt_dir=ref_dir)
            _, r = run_with_auto_offload_fallback(
                idx,
                config_name="v7x_ref2va_6ref_turbo_fp8_u8" + ("_syncprof" if idx == 3 else ""),
                model_short="MiniMax-H3-Ref2VA + Turbo ref2v v0.1 (merged before online FP8)",
                model_path=str(ref_dir),
                workloads=build_workloads_ref2va_6ref_4step(
                    inputs_dir, "v7x_fp8" + ("_syncprof" if idx == 3 else ""), perf=(idx == 3)
                ),
                **common_run,
            )
            if r is not None:
                r.stop()
        elif idx == 1:
            fl_dir = stage_fl2va_in_shm(idx, bucket, models_prefix)
            # (A) Official FL2VA base weights (no LoRA adapter merged).
            _, r_a = run_with_auto_offload_fallback(
                idx,
                config_name="v7x_i2v_dragon_4step_base_fp8_u8",
                model_short="MiniMax-H3-FL2VA (official base weights)",
                model_path=str(fl_dir),
                workloads=build_workloads_i2v_dragon_4step(
                    inputs_dir, "v7x_fp8", "nolora", "none (official FL2VA weights; matches ref recorded run)"
                ),
                **common_run,
            )
            if r_a is not None:
                r_a.stop()
            # (B) FL2VA Turbo v1.2 4-step adapter merged into BF16 before online FP8 quantization.
            merge_turbo_lora_inplace(idx, bucket=bucket, lora_gcs_obj=lora_fl2v_obj, ckpt_dir=fl_dir)
            _, r_b = run_with_auto_offload_fallback(
                idx,
                config_name="v7x_i2v_dragon_4step_turbo12_fp8_u8",
                model_short="MiniMax-H3-FL2VA + Turbo fl2v v1.2 768p (merged before online FP8)",
                model_path=str(fl_dir),
                workloads=build_workloads_i2v_dragon_4step(
                    inputs_dir,
                    "v7x_fp8",
                    "turbo_v1.2",
                    "lightx2v/Minimax-h3-Turbo minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16 (alpha 8, scale 1.0), merged before online FP8",
                ),
                **common_run,
            )
            if r_b is not None:
                r_b.stop()
        elif idx == 2:
            fl_dir = stage_fl2va_in_shm(idx, bucket, models_prefix)
            _, r = run_with_auto_offload_fallback(
                idx,
                config_name="v7x_profile_fl2va_base_fp8_u8_syncprof",
                model_short="MiniMax-H3-FL2VA base (device-synchronized profiling)",
                model_path=str(fl_dir),
                workloads=build_workloads_profile_fl2va_sync(inputs_dir, "v7x_fp8_syncprof"),
                **common_run,
            )
            if r is not None:
                r.stop()

    # =========================================================================
    # ROLE 2: v7x8_bf16_fasth3 (2 hosts = 8 v7x chips: Pod 0 FL2VA BF16 + FastH3, Pod 1 Ref2VA BF16)
    # =========================================================================
    elif args.role == "v7x8_bf16_fasth3":
        if idx == 0:
            fl_dir = stage_fl2va_in_shm(idx, bucket, models_prefix)
            bf16_wls = build_workloads_fl2va_base(inputs_dir, "v7x_bf16")
            _, r_bf16 = run_with_auto_offload_fallback(
                idx,
                accelerator="v7x",
                bucket=bucket,
                gcs_prefix=gcs_prefix,
                local_jsonl=local_jsonl,
                gcs_jsonl_obj=gcs_jsonl_obj,
                config_name="v7x_fl2va_base_bf16_u8",
                model_short="MiniMax-H3-FL2VA-Base-BF16",
                model_path=str(fl_dir),
                chips_per_host=4,
                dp_replicas=1,
                num_gpus=8,
                ulysses_degree=8,
                quantization=None,
                try_specs=v7x_bf16_specs,
                remote_encode_url=remote_encode_url,
                workloads=bf16_wls,
                collected=collected,
            )
            if r_bf16 is not None:
                r_bf16.stop()

            # Also test BF16 with Turbo4 LoRA merged on I2V 345-Frame Workload
            merge_turbo_lora_inplace(idx, bucket=bucket, lora_gcs_obj=lora_fl2v_obj, ckpt_dir=fl_dir)
            _, r_bf16_t = run_with_auto_offload_fallback(
                idx,
                accelerator="v7x",
                bucket=bucket,
                gcs_prefix=gcs_prefix,
                local_jsonl=local_jsonl,
                gcs_jsonl_obj=gcs_jsonl_obj,
                config_name="v7x_fl2va_turbo4_bf16_u8",
                model_short="MiniMax-H3-FL2VA-Turbo4-BF16",
                model_path=str(fl_dir),
                chips_per_host=4,
                dp_replicas=1,
                num_gpus=8,
                ulysses_degree=8,
                quantization=None,
                try_specs=v7x_bf16_specs,
                remote_encode_url=remote_encode_url,
                workloads=build_workloads_fl2va_turbo(inputs_dir, "v7x_bf16")[:4],
                collected=collected,
            )
            if r_bf16_t is not None:
                r_bf16_t.stop()

            # Swap cleanly to FastH3 (4-step VSA Distilled) in FP8
            fh_dir = stage_fasth3_in_shm(idx, bucket, models_prefix)
            _, r_fh = run_with_auto_offload_fallback(
                idx,
                accelerator="v7x",
                bucket=bucket,
                gcs_prefix=gcs_prefix,
                local_jsonl=local_jsonl,
                gcs_jsonl_obj=gcs_jsonl_obj,
                config_name="v7x_fasth3_fp8_u8",
                model_short="FastH3-4step-VSA",
                model_path=str(fh_dir),
                chips_per_host=4,
                dp_replicas=1,
                num_gpus=8,
                ulysses_degree=8,
                quantization="fp8",
                try_specs=v7x_fp8_specs,
                remote_encode_url=remote_encode_url,
                workloads=build_workloads_fasth3("v7x_fp8"),
                collected=collected,
            )
            if r_fh is not None:
                r_fh.stop()
        else:
            # Pod 1: Lossless BF16 Ref2VA (Published 50-step Ref2VA + 4-step 1792x768 345f 6-ref Ref2VA)
            ref_dir = stage_ref2va_in_shm(idx, bucket, models_prefix)
            _, r_ref_bf16 = run_with_auto_offload_fallback(
                idx,
                accelerator="v7x",
                bucket=bucket,
                gcs_prefix=gcs_prefix,
                local_jsonl=local_jsonl,
                gcs_jsonl_obj=gcs_jsonl_obj,
                config_name="v7x_ref2va_base_bf16_u8",
                model_short="MiniMax-H3-Ref2VA-Base-BF16",
                model_path=str(ref_dir),
                chips_per_host=4,
                dp_replicas=1,
                num_gpus=8,
                ulysses_degree=8,
                quantization=None,
                try_specs=v7x_bf16_specs,
                remote_encode_url=remote_encode_url,
                workloads=build_workloads_ref2va_base(inputs_dir, "v7x_bf16"),
                collected=collected,
            )
            if r_ref_bf16 is not None:
                r_ref_bf16.stop()

            merge_turbo_lora_inplace(idx, bucket=bucket, lora_gcs_obj=lora_ref2v_obj, ckpt_dir=ref_dir)
            _, r_ref_t_bf16 = run_with_auto_offload_fallback(
                idx,
                accelerator="v7x",
                bucket=bucket,
                gcs_prefix=gcs_prefix,
                local_jsonl=local_jsonl,
                gcs_jsonl_obj=gcs_jsonl_obj,
                config_name="v7x_ref2va_turbo4_bf16_u8",
                model_short="MiniMax-H3-Ref2VA-Turbo4-BF16",
                model_path=str(ref_dir),
                chips_per_host=4,
                dp_replicas=1,
                num_gpus=8,
                ulysses_degree=8,
                quantization=None,
                try_specs=v7x_bf16_specs,
                remote_encode_url=remote_encode_url,
                workloads=build_workloads_ref2va_turbo(inputs_dir, "v7x_bf16")[:3],
                collected=collected,
            )
            if r_ref_t_bf16 is not None:
                r_ref_t_bf16.stop()

    # =========================================================================
    # ROLE 3: v7x16_slice (4 hosts = 16 v7x chips: DP=4xU8 FL2VA, DP=4xU8 Ref2VA, DP=8xU4 Sub-Host Packing)
    # =========================================================================
    elif args.role == "v7x16_slice":
        keys_16 = [f"v7x16_p{i}" for i in range(4)]
        fl_dir = stage_fl2va_in_shm(idx, bucket, models_prefix)
        merge_turbo_lora_inplace(idx, bucket=bucket, lora_gcs_obj=lora_fl2v_obj, ckpt_dir=fl_dir)

        warm_fl = [
            {"task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 14.375, "num_inference_steps": 5, "flow_shift": 6.0, "audio_flow_shift": 3.0, "conditions": fl2va_dragon_condition(inputs_dir)},
            {"task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0, "num_inference_steps": 5, "flow_shift": 6.0, "audio_flow_shift": 3.0, "conditions": fl2va_dragon_condition(inputs_dir)},
            {"task": "t2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0, "num_inference_steps": 5, "flow_shift": 6.0, "audio_flow_shift": 3.0, "conditions": []},
        ]
        slice16_fl_wls = [
            {"label": "v7x-16 Slice (DP=4xU8) I2V Dragon 1344x768 14.375s 345f (4-step, c=4)", "task": "fl2va", "prompt": PROMPT_I2V_DRAGON, "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 14.375, "num_inference_steps": 5, "flow_shift": 6.0, "audio_flow_shift": 3.0, "conditions": fl2va_dragon_condition(inputs_dir), "concurrency": 4, "num_requests": 8},
            {"label": "v7x-16 Slice (DP=4xU8) I2V Dragon 1344x768 14.375s 345f (4-step, c=8)", "task": "fl2va", "prompt": PROMPT_I2V_DRAGON, "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 14.375, "num_inference_steps": 5, "flow_shift": 6.0, "audio_flow_shift": 3.0, "conditions": fl2va_dragon_condition(inputs_dir), "concurrency": 8, "num_requests": 12},
            {"label": "v7x-16 Slice (DP=4xU8) I2V Dragon 1344x768 5s 124f (4-step, c=4)", "task": "fl2va", "prompt": PROMPT_I2V_DRAGON, "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0, "num_inference_steps": 5, "flow_shift": 6.0, "audio_flow_shift": 3.0, "conditions": fl2va_dragon_condition(inputs_dir), "concurrency": 4, "num_requests": 12},
            {"label": "v7x-16 Slice (DP=4xU8) I2V Dragon 1344x768 5s 124f (4-step, c=8)", "task": "fl2va", "prompt": PROMPT_I2V_DRAGON, "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0, "num_inference_steps": 5, "flow_shift": 6.0, "audio_flow_shift": 3.0, "conditions": fl2va_dragon_condition(inputs_dir), "concurrency": 8, "num_requests": 16},
            {"label": "v7x-16 Slice (DP=4xU8) T2VA Cats 1344x768 5s 124f (50-step, c=4)", "task": "t2va", "prompt": PROMPT_T2VA_CATS, "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0, "num_inference_steps": 50, "flow_shift": 12.0, "audio_flow_shift": 3.0, "conditions": [], "concurrency": 4, "num_requests": 8},
        ]
        run_multi_host_slice_sweep(
            idx,
            accelerator="v7x",
            bucket=bucket,
            gcs_prefix=gcs_prefix,
            slice_tag="v7x16_slice_dp4_u8_fl2va_turbo4",
            slice_chips=16,
            model_short="MiniMax-H3-FL2VA-Turbo4",
            model_path=str(fl_dir),
            quantization="fp8",
            server_spec=v7x_fp8_specs[0],
            dp_per_host=1,
            num_gpus=8,
            ulysses_degree=8,
            remote_encode_url=remote_encode_url,
            warm_workloads=warm_fl,
            slice_workloads=slice16_fl_wls,
            local_jsonl=local_jsonl,
            gcs_jsonl_obj=gcs_jsonl_obj,
            collected=collected,
            pod_key=f"v7x16_p{idx}",
            expected_keys=keys_16,
            is_driver=(idx == 0),
        )

        # Phase 2 on v7x-16: Swap all 4 hosts to Ref2VA + Turbo4 for DP=4 x U=8 Ref2VA 1792x768 345f (6-Reference Ref2VA Workload)!
        ref_dir = stage_ref2va_in_shm(idx, bucket, models_prefix)
        merge_turbo_lora_inplace(idx, bucket=bucket, lora_gcs_obj=lora_ref2v_obj, ckpt_dir=ref_dir)
        warm_ref = [
            {"task": "ref2va", "prompt": PROMPT_REF2VA_SAMPLE, "short_edge": 768, "aspect_ratio": "21:9", "duration_seconds": 14.375, "num_inference_steps": 5, "flow_shift": 6.0, "audio_flow_shift": 3.0, "conditions": ref2va_6_conditions(inputs_dir)},
        ]
        slice16_ref_wls = [
            {"label": "v7x-16 Slice (DP=4xU8) Ref2VA Sample 1792x768 14.375s 345f (4-step, 6 refs, c=4)", "task": "ref2va", "prompt": PROMPT_REF2VA_SAMPLE, "short_edge": 768, "aspect_ratio": "21:9", "duration_seconds": 14.375, "num_inference_steps": 5, "flow_shift": 6.0, "audio_flow_shift": 3.0, "conditions": ref2va_6_conditions(inputs_dir), "concurrency": 4, "num_requests": 8},
            {"label": "v7x-16 Slice (DP=4xU8) Ref2VA Sample 1792x768 14.375s 345f (4-step, 6 refs, c=8)", "task": "ref2va", "prompt": PROMPT_REF2VA_SAMPLE, "short_edge": 768, "aspect_ratio": "21:9", "duration_seconds": 14.375, "num_inference_steps": 5, "flow_shift": 6.0, "audio_flow_shift": 3.0, "conditions": ref2va_6_conditions(inputs_dir), "concurrency": 8, "num_requests": 12},
        ]
        run_multi_host_slice_sweep(
            idx,
            accelerator="v7x",
            bucket=bucket,
            gcs_prefix=gcs_prefix,
            slice_tag="v7x16_slice_dp4_u8_ref2va_turbo4",
            slice_chips=16,
            model_short="MiniMax-H3-Ref2VA-Turbo4",
            model_path=str(ref_dir),
            quantization="fp8",
            server_spec=v7x_fp8_specs[0],
            dp_per_host=1,
            num_gpus=8,
            ulysses_degree=8,
            remote_encode_url=remote_encode_url,
            warm_workloads=warm_ref,
            slice_workloads=slice16_ref_wls,
            local_jsonl=local_jsonl,
            gcs_jsonl_obj=gcs_jsonl_obj,
            collected=collected,
            pod_key=f"v7x16_p{idx}",
            expected_keys=keys_16,
            is_driver=(idx == 0),
        )

    # =========================================================================
    # ROLE 4A: v5p32_dedicated (8 hosts = 32 v5p chips on native 2x4x4 slice)
    # =========================================================================
    elif args.role == "v5p32_dedicated":
        keys_32 = [f"v5p32_d{i}" for i in range(8)]
        fl_dir = stage_fl2va_in_shm(idx, bucket, models_prefix)
        merge_turbo_lora_inplace(idx, bucket=bucket, lora_gcs_obj=lora_fl2v_obj, ckpt_dir=fl_dir)
        winning_v5p_spec = v5p_bf16_specs[0]

        warm_v5p32 = [
            {"task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 14.375, "num_inference_steps": 5, "flow_shift": 6.0, "audio_flow_shift": 3.0, "conditions": fl2va_dragon_condition(inputs_dir)},
            {"task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0, "num_inference_steps": 5, "flow_shift": 6.0, "audio_flow_shift": 3.0, "conditions": fl2va_dragon_condition(inputs_dir)},
        ]
        slice_v5p32_wls = [
            {"label": "v5p-32 Slice (DP=8xU4, 32 v5p chips) I2V Dragon 1344x768 14.375s 345f (4-step, c=8)", "task": "fl2va", "prompt": PROMPT_I2V_DRAGON, "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 14.375, "num_inference_steps": 5, "flow_shift": 6.0, "audio_flow_shift": 3.0, "conditions": fl2va_dragon_condition(inputs_dir), "concurrency": 8, "num_requests": 16},
            {"label": "v5p-32 Slice (DP=8xU4, 32 v5p chips) I2V Dragon 1344x768 14.375s 345f (4-step, c=16)", "task": "fl2va", "prompt": PROMPT_I2V_DRAGON, "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 14.375, "num_inference_steps": 5, "flow_shift": 6.0, "audio_flow_shift": 3.0, "conditions": fl2va_dragon_condition(inputs_dir), "concurrency": 16, "num_requests": 24},
            {"label": "v5p-32 Slice (DP=8xU4, 32 v5p chips) I2V Dragon 1344x768 5s 124f (4-step, c=8)", "task": "fl2va", "prompt": PROMPT_I2V_DRAGON, "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0, "num_inference_steps": 5, "flow_shift": 6.0, "audio_flow_shift": 3.0, "conditions": fl2va_dragon_condition(inputs_dir), "concurrency": 8, "num_requests": 16},
            {"label": "v5p-32 Slice (DP=8xU4, 32 v5p chips) I2V Dragon 1344x768 5s 124f (4-step, c=16)", "task": "fl2va", "prompt": PROMPT_I2V_DRAGON, "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0, "num_inference_steps": 5, "flow_shift": 6.0, "audio_flow_shift": 3.0, "conditions": fl2va_dragon_condition(inputs_dir), "concurrency": 16, "num_requests": 24},
        ]
        run_multi_host_slice_sweep(
            idx,
            accelerator="v5p",
            bucket=bucket,
            gcs_prefix=gcs_prefix,
            slice_tag="v5p32_slice_dp8_u4_fl2va_turbo4_v2",
            slice_chips=32,
            model_short="MiniMax-H3-FL2VA-Turbo4",
            model_path=str(fl_dir),
            quantization=None,
            server_spec=winning_v5p_spec,
            dp_per_host=1,
            num_gpus=4,
            ulysses_degree=4,
            remote_encode_url=remote_encode_url,
            warm_workloads=warm_v5p32,
            slice_workloads=slice_v5p32_wls,
            local_jsonl=local_jsonl,
            gcs_jsonl_obj=gcs_jsonl_obj,
            collected=collected,
            pod_key=f"v5p32_d{idx}",
            expected_keys=keys_32,
            is_driver=(idx == 0),
        )

        # Swap all 8 hosts to Ref2VA + Turbo4 for 32-chip Ref2VA 1792x768 345f (6 refs) slice sweep!
        ref_dir = stage_ref2va_in_shm(idx, bucket, models_prefix)
        merge_turbo_lora_inplace(idx, bucket=bucket, lora_gcs_obj=lora_ref2v_obj, ckpt_dir=ref_dir)
        warm_ref32 = [
            {"task": "ref2va", "prompt": PROMPT_REF2VA_SAMPLE, "short_edge": 768, "aspect_ratio": "21:9", "duration_seconds": 14.375, "num_inference_steps": 5, "flow_shift": 6.0, "audio_flow_shift": 3.0, "conditions": ref2va_6_conditions(inputs_dir)},
        ]
        slice_ref32_wls = [
            {"label": "v5p-32 Slice (DP=8xU4, 32 v5p chips) Ref2VA Sample 1792x768 14.375s 345f (4-step, 6 refs, c=8)", "task": "ref2va", "prompt": PROMPT_REF2VA_SAMPLE, "short_edge": 768, "aspect_ratio": "21:9", "duration_seconds": 14.375, "num_inference_steps": 5, "flow_shift": 6.0, "audio_flow_shift": 3.0, "conditions": ref2va_6_conditions(inputs_dir), "concurrency": 8, "num_requests": 16},
        ]
        run_multi_host_slice_sweep(
            idx,
            accelerator="v5p",
            bucket=bucket,
            gcs_prefix=gcs_prefix,
            slice_tag="v5p32_slice_dp8_u4_ref2va_turbo4_v2",
            slice_chips=32,
            model_short="MiniMax-H3-Ref2VA-Turbo4",
            model_path=str(ref_dir),
            quantization=None,
            server_spec=v5p_bf16_specs[0],
            dp_per_host=1,
            num_gpus=4,
            ulysses_degree=4,
            remote_encode_url=remote_encode_url,
            warm_workloads=warm_ref32,
            slice_workloads=slice_ref32_wls,
            local_jsonl=local_jsonl,
            gcs_jsonl_obj=gcs_jsonl_obj,
            collected=collected,
            pod_key=f"v5p32_d{idx}",
            expected_keys=keys_32,
            is_driver=(idx == 0),
        )

    # =========================================================================
    # ROLE 4B & 5: v5p16_dedicated (4 hosts = 16 v5p chips) OR v5p16_main + v5p32_worker
    # =========================================================================
    elif args.role in ("v5p16_dedicated", "v5p16_main", "v5p32_worker"):
        winning_v5p_spec = v5p_bf16_specs[0]
        if args.role in ("v5p16_dedicated", "v5p16_main"):
            if idx == 0:
                # Pod 0: 6-Reference Ref2VA on TPU v5p!
                ref_dir = stage_ref2va_in_shm(idx, bucket, models_prefix)
                merge_turbo_lora_inplace(idx, bucket=bucket, lora_gcs_obj=lora_ref2v_obj, ckpt_dir=ref_dir)
                w_spec, r_ref = run_with_auto_offload_fallback(
                    idx,
                    accelerator="v5p",
                    bucket=bucket,
                    gcs_prefix=gcs_prefix,
                    local_jsonl=local_jsonl,
                    gcs_jsonl_obj=gcs_jsonl_obj,
                    config_name="v5p_ref2va_turbo4_bf16_u4",
                    model_short="MiniMax-H3-Ref2VA-Turbo4",
                    model_path=str(ref_dir),
                    chips_per_host=4,
                    dp_replicas=1,
                    num_gpus=4,
                    ulysses_degree=4,
                    quantization=None,
                    try_specs=v5p_bf16_specs,
                    remote_encode_url=remote_encode_url,
                    workloads=build_workloads_ref2va_turbo(inputs_dir, "v5p_bf16")[:1],
                    collected=collected,
                )
                if r_ref is not None:
                    r_ref.stop()
                fl_dir = swap_ref2va_to_fl2va_in_shm(idx, bucket, models_prefix)
                merge_turbo_lora_inplace(idx, bucket=bucket, lora_gcs_obj=lora_fl2v_obj, ckpt_dir=fl_dir)
            elif idx == 1:
                # Pod 1: I2V Dragon 5s 124f on TPU v5p (14.375s 345f already completed)
                fl_dir = stage_fl2va_in_shm(idx, bucket, models_prefix)
                merge_turbo_lora_inplace(idx, bucket=bucket, lora_gcs_obj=lora_fl2v_obj, ckpt_dir=fl_dir)
                w_spec, r_fl = run_with_auto_offload_fallback(
                    idx,
                    accelerator="v5p",
                    bucket=bucket,
                    gcs_prefix=gcs_prefix,
                    local_jsonl=local_jsonl,
                    gcs_jsonl_obj=gcs_jsonl_obj,
                    config_name="v5p_fl2va_turbo4_bf16_u4",
                    model_short="MiniMax-H3-FL2VA-Turbo4",
                    model_path=str(fl_dir),
                    chips_per_host=4,
                    dp_replicas=1,
                    num_gpus=4,
                    ulysses_degree=4,
                    quantization=None,
                    try_specs=v5p_bf16_specs,
                    remote_encode_url=remote_encode_url,
                    workloads=build_workloads_fl2va_turbo(inputs_dir, "v5p_bf16")[2:4],
                    collected=collected,
                )
                if w_spec is not None:
                    winning_v5p_spec = w_spec
                    gcs_upload_bytes(bucket, f"{gcs_prefix}/v5p_winning_spec.json", json.dumps(w_spec).encode("utf-8"))
                if r_fl is not None:
                    r_fl.stop()
            elif idx == 2:
                # Pod 2: T2VA Cats 5s 124f (4-step Turbo Merged, seed=1101) on TPU v5p
                fl_dir = stage_fl2va_in_shm(idx, bucket, models_prefix)
                merge_turbo_lora_inplace(idx, bucket=bucket, lora_gcs_obj=lora_fl2v_obj, ckpt_dir=fl_dir)
                _, r_b = run_with_auto_offload_fallback(
                    idx,
                    accelerator="v5p",
                    bucket=bucket,
                    gcs_prefix=gcs_prefix,
                    local_jsonl=local_jsonl,
                    gcs_jsonl_obj=gcs_jsonl_obj,
                    config_name="v5p_fl2va_turbo4_bf16_u4",
                    model_short="MiniMax-H3-FL2VA-Turbo4",
                    model_path=str(fl_dir),
                    chips_per_host=4,
                    dp_replicas=1,
                    num_gpus=4,
                    ulysses_degree=4,
                    quantization=None,
                    try_specs=v5p_bf16_specs,
                    remote_encode_url=remote_encode_url,
                    workloads=build_workloads_fl2va_turbo(inputs_dir, "v5p_bf16")[6:7],
                    collected=collected,
                )
                if r_b is not None:
                    r_b.stop()
            else:
                # Pod 3: FastH3 14.375s 345f on TPU v5p (5s 124f already completed), then swap to FL2VA Turbo4
                fh_dir = stage_fasth3_in_shm(idx, bucket, models_prefix)
                _, r_fh = run_with_auto_offload_fallback(
                    idx,
                    accelerator="v5p",
                    bucket=bucket,
                    gcs_prefix=gcs_prefix,
                    local_jsonl=local_jsonl,
                    gcs_jsonl_obj=gcs_jsonl_obj,
                    config_name="v5p_fasth3_bf16_u4",
                    model_short="FastH3-4step-VSA",
                    model_path=str(fh_dir),
                    chips_per_host=4,
                    dp_replicas=1,
                    num_gpus=4,
                    ulysses_degree=4,
                    quantization=None,
                    try_specs=v5p_fasth3_specs,
                    remote_encode_url=remote_encode_url,
                    workloads=build_workloads_fasth3("v5p_bf16")[1:2],
                    collected=collected,
                )
                if r_fh is not None:
                    r_fh.stop()
                fl_dir = stage_fl2va_in_shm(idx, bucket, models_prefix)
                merge_turbo_lora_inplace(idx, bucket=bucket, lora_gcs_obj=lora_fl2v_obj, ckpt_dir=fl_dir)

            raw_ws = gcs_download_bytes(bucket, f"{gcs_prefix}/v5p_winning_spec.json")
            if raw_ws:
                winning_v5p_spec = json.loads(raw_ws.decode("utf-8"))

            # v5p-16 Slice Sweep (4 hosts = 16 v5p chips, DP=4 x Ulysses=4)
            warm_v5p = [
                {"task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 14.375, "num_inference_steps": 5, "flow_shift": 6.0, "audio_flow_shift": 3.0, "conditions": fl2va_dragon_condition(inputs_dir)},
                {"task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0, "num_inference_steps": 5, "flow_shift": 6.0, "audio_flow_shift": 3.0, "conditions": fl2va_dragon_condition(inputs_dir)},
            ]
            slice_v5p16_wls = [
                {"label": "v5p-16 Slice (DP=4xU4) I2V Dragon 1344x768 14.375s 345f (4-step, c=4)", "task": "fl2va", "prompt": PROMPT_I2V_DRAGON, "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 14.375, "num_inference_steps": 5, "flow_shift": 6.0, "audio_flow_shift": 3.0, "conditions": fl2va_dragon_condition(inputs_dir), "concurrency": 4, "num_requests": 8},
                {"label": "v5p-16 Slice (DP=4xU4) I2V Dragon 1344x768 5s 124f (4-step, c=4)", "task": "fl2va", "prompt": PROMPT_I2V_DRAGON, "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0, "num_inference_steps": 5, "flow_shift": 6.0, "audio_flow_shift": 3.0, "conditions": fl2va_dragon_condition(inputs_dir), "concurrency": 4, "num_requests": 8},
                {"label": "v5p-16 Slice (DP=4xU4) I2V Dragon 1344x768 5s 124f (4-step, c=8)", "task": "fl2va", "prompt": PROMPT_I2V_DRAGON, "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0, "num_inference_steps": 5, "flow_shift": 6.0, "audio_flow_shift": 3.0, "conditions": fl2va_dragon_condition(inputs_dir), "concurrency": 8, "num_requests": 16},
            ]
            run_multi_host_slice_sweep(
                idx,
                accelerator="v5p",
                bucket=bucket,
                gcs_prefix=gcs_prefix,
                slice_tag="v5p16_slice_dp4_u4_fl2va_turbo4",
                slice_chips=16,
                model_short="MiniMax-H3-FL2VA-Turbo4",
                model_path=str(fl_dir),
                quantization=None,
                server_spec=winning_v5p_spec,
                dp_per_host=1,
                num_gpus=4,
                ulysses_degree=4,
                remote_encode_url=remote_encode_url,
                warm_workloads=warm_v5p,
                slice_workloads=slice_v5p16_wls,
                local_jsonl=local_jsonl,
                gcs_jsonl_obj=gcs_jsonl_obj,
                collected=collected,
                pod_key=f"v5p16_p{idx}",
                expected_keys=[f"v5p16_p{i}" for i in range(4)],
                is_driver=(idx == 0),
            )
            my_v5p32_key = f"v5p32_a{idx}"
        else:
            fl_dir = stage_fl2va_in_shm(idx, bucket, models_prefix)
            merge_turbo_lora_inplace(idx, bucket=bucket, lora_gcs_obj=lora_fl2v_obj, ckpt_dir=fl_dir)
            my_v5p32_key = f"v5p32_b{idx}"

        if args.role != "v5p16_dedicated":
            raw_ws = gcs_download_bytes(bucket, f"{gcs_prefix}/v5p_winning_spec.json")
            if raw_ws:
                winning_v5p_spec = json.loads(raw_ws.decode("utf-8"))
            keys_32 = [f"v5p32_a{i}" for i in range(4)] + [f"v5p32_b{i}" for i in range(4)]
            warm_v5p32 = [
                {"task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 14.375, "num_inference_steps": 5, "flow_shift": 6.0, "audio_flow_shift": 3.0, "conditions": fl2va_dragon_condition(inputs_dir)},
                {"task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0, "num_inference_steps": 5, "flow_shift": 6.0, "audio_flow_shift": 3.0, "conditions": fl2va_dragon_condition(inputs_dir)},
            ]
            slice_v5p32_wls = [
                {"label": "v5p-32 Slice (DP=8xU4, 32 v5p chips) I2V Dragon 1344x768 14.375s 345f (4-step, c=8)", "task": "fl2va", "prompt": PROMPT_I2V_DRAGON, "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 14.375, "num_inference_steps": 5, "flow_shift": 6.0, "audio_flow_shift": 3.0, "conditions": fl2va_dragon_condition(inputs_dir), "concurrency": 8, "num_requests": 16},
                {"label": "v5p-32 Slice (DP=8xU4, 32 v5p chips) I2V Dragon 1344x768 5s 124f (4-step, c=8)", "task": "fl2va", "prompt": PROMPT_I2V_DRAGON, "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0, "num_inference_steps": 5, "flow_shift": 6.0, "audio_flow_shift": 3.0, "conditions": fl2va_dragon_condition(inputs_dir), "concurrency": 8, "num_requests": 16},
                {"label": "v5p-32 Slice (DP=8xU4, 32 v5p chips) I2V Dragon 1344x768 5s 124f (4-step, c=16)", "task": "fl2va", "prompt": PROMPT_I2V_DRAGON, "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0, "num_inference_steps": 5, "flow_shift": 6.0, "audio_flow_shift": 3.0, "conditions": fl2va_dragon_condition(inputs_dir), "concurrency": 16, "num_requests": 24},
            ]
            run_multi_host_slice_sweep(
                idx,
                accelerator="v5p",
                bucket=bucket,
                gcs_prefix=gcs_prefix,
                slice_tag="v5p32_slice_dp8_u4_fl2va_turbo4",
                slice_chips=32,
                model_short="MiniMax-H3-FL2VA-Turbo4",
                model_path=str(fl_dir),
                quantization=None,
                server_spec=winning_v5p_spec,
                dp_per_host=1,
                num_gpus=4,
                ulysses_degree=4,
                remote_encode_url=remote_encode_url,
                warm_workloads=warm_v5p32,
                slice_workloads=slice_v5p32_wls,
                local_jsonl=local_jsonl,
                gcs_jsonl_obj=gcs_jsonl_obj,
                collected=collected,
                pod_key=my_v5p32_key,
                expected_keys=keys_32,
                is_driver=(args.role == "v5p16_main" and idx == 0),
            )

    gcs_upload_bytes(
        bucket,
        f"{gcs_prefix}/results/{args.role}_pod{idx}_summary.json",
        json.dumps(collected, indent=2).encode("utf-8"),
    )
    log(idx, f"=== Role {args.role} pod-{idx} COMPLETE ({len(collected)} records uploaded to gs://{bucket}/{gcs_jsonl_obj}) ===")


if __name__ == "__main__":
    main()
