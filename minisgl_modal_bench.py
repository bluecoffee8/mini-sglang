from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

import modal

APP = modal.App("minisgl-bench-qwen")
REMOTE_ROOT = "/root/mini-sglang"
CACHE_ROOT = "/mnt/mini-sglang-cache2"
CACHE = modal.Volume.from_name("mini-sglang-cache2")

MODEL = "Qwen/Qwen3-8B"
PORT = 1919
SERVER_STARTUP_TIMEOUT_S = 30 * 60

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "libnuma1")
    .pip_install("uv")
    .run_commands(
        "git clone --branch n_gram --depth 1 "
        "https://github.com/bluecoffee8/mini-sglang.git /root/mini-sglang",
        f"cd {REMOTE_ROOT} && uv venv --python=3.12",
        f"cd {REMOTE_ROOT} && . .venv/bin/activate && uv pip install -e .",
    )
)


def _configure_cache_env() -> None:
    os.environ["HF_HOME"] = f"{CACHE_ROOT}/huggingface"
    os.environ["XDG_CACHE_HOME"] = CACHE_ROOT
    os.environ["FLASHINFER_WORKSPACE_BASE"] = f"{CACHE_ROOT}/flashinfer"
    os.environ["TVM_FFI_CACHE_DIR"] = f"{CACHE_ROOT}/tvm-ffi"
    os.environ["TORCH_EXTENSIONS_DIR"] = f"{CACHE_ROOT}/torch_extensions"


def _wait_for_server(proc: subprocess.Popen, port: int, timeout_s: float) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + timeout_s
    last_log = 0.0
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"minisgl server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    print("minisgl server is ready")
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        if time.time() - last_log > 30:
            print("waiting for minisgl server to become ready...")
            last_log = time.time()
        time.sleep(2)
    raise TimeoutError(f"minisgl server did not become ready within {timeout_s:.0f}s")


def _sync_to_latest_commit() -> str:
    # The image's git clone is baked in at image-build time and then cached by
    # Modal, so it can silently go stale relative to the branch on GitHub. Re-fetch
    # and switch to the branch tip at call time so we always run the latest commit.
    subprocess.run(
        ["git", "fetch", "--depth", "1", "origin", "n_gram"],
        cwd=REMOTE_ROOT,
        check=True,
    )
    subprocess.run(
        ["git", "switch", "--detach", "FETCH_HEAD"],
        cwd=REMOTE_ROOT,
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REMOTE_ROOT,
        text=True,
    ).strip()
    print(f"synced {REMOTE_ROOT} to n_gram commit {commit}")
    return commit


@APP.function(image=image, gpu="H100!", timeout=2 * 60 * 60, volumes={CACHE_ROOT: CACHE})
def run_benchmark(model: str = MODEL) -> str:
    _sync_to_latest_commit()
    _configure_cache_env()
    env = os.environ.copy()
    env["PATH"] = f"{REMOTE_ROOT}/.venv/bin:" + env["PATH"]
    python_bin = f"{REMOTE_ROOT}/.venv/bin/python"

    server_cmd = [
        python_bin,
        "-m",
        "minisgl",
        "--model",
        model,
        "--port",
        str(PORT),
        # "auto" resolves to hybrid "fa,fi" on H100 (sm90), and importing the "fa"
        # backend's sgl_kernel.flash_attn module unconditionally pulls in a
        # flash_attn_origin.cute/cutlass-dsl import chain that crashes at import
        # time in this environment (cutlass/sgl_kernel version mismatch), killing
        # the scheduler subprocess silently. Force flashinfer to avoid that path.
        "--attention-backend",
        "fi",
    ]    
    print(f"starting minisgl server: {' '.join(server_cmd)}")
    server = subprocess.Popen(server_cmd, cwd=REMOTE_ROOT, env=env)
    try:
        _wait_for_server(server, PORT, SERVER_STARTUP_TIMEOUT_S)

        print("starting benchmark/online/bench_qwen.py")
        client = subprocess.run(
            [python_bin, "benchmark/online/bench_qwen.py"],
            cwd=REMOTE_ROOT,
            env=env,
        )
        if client.returncode:
            raise RuntimeError(f"bench_qwen.py failed with exit code {client.returncode}")
        return "benchmark completed successfully"
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


@APP.local_entrypoint()
def main(model: str = MODEL) -> None:
    print(run_benchmark.remote(model))

# modal run minisgl_modal_bench.py
