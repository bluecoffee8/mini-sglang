from __future__ import annotations

import os
import subprocess

import modal

APP = modal.App("minisgl-check-ngram")
REMOTE_ROOT = "/root/mini-sglang"
CACHE_ROOT = "/mnt/mini-sglang-cache2"
CACHE = modal.Volume.from_name("mini-sglang-cache2")

MODEL = "Qwen/Qwen3-8B"
CHECK_SCRIPT = "benchmark/offline/check_ngram_correctness.py"

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


@APP.function(image=image, gpu="H100!", timeout=60 * 60, volumes={CACHE_ROOT: CACHE})
def run_check(model: str = MODEL) -> str:
    _sync_to_latest_commit()
    _configure_cache_env()
    env = os.environ.copy()
    env["PATH"] = f"{REMOTE_ROOT}/.venv/bin:" + env["PATH"]
    python_bin = f"{REMOTE_ROOT}/.venv/bin/python"

    baseline_out = "/tmp/ngram_check_baseline.json"
    speculative_out = "/tmp/ngram_check_speculative.json"

    def _run_pass(mode: str, output: str) -> None:
        # Each pass is its own subprocess: the engine's global context
        # (minisgl.core._GLOBAL_CTX) can only be initialized once per process, so a
        # baseline run and a speculative run can't share one Python process.
        print(f"=== running {mode} pass ===")
        result = subprocess.run(
            [python_bin, CHECK_SCRIPT, "--model", model, "run", "--mode", mode, "--output", output],
            cwd=REMOTE_ROOT,
            env=env,
        )
        if result.returncode:
            raise RuntimeError(f"{mode} pass failed with exit code {result.returncode}")

    _run_pass("baseline", baseline_out)
    _run_pass("speculative", speculative_out)

    print("=== comparing baseline vs speculative token ids ===")
    compare_result = subprocess.run(
        [python_bin, CHECK_SCRIPT, "compare", baseline_out, speculative_out],
        cwd=REMOTE_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    print(compare_result.stdout)
    if compare_result.stderr:
        print(compare_result.stderr)
    if compare_result.returncode:
        raise RuntimeError(
            "n-gram speculative decoding correctness check FAILED: greedy output "
            "diverged between baseline and speculative runs (see comparison above)."
        )
    return "n-gram speculative decoding correctness check PASSED"


@APP.local_entrypoint()
def main(model: str = MODEL) -> None:
    print(run_check.remote(model))

# modal run minisgl_modal_check_ngram.py
