from __future__ import annotations

import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import modal


APP = modal.App("minisgl-ngram2")

CACHE_ROOT = "/mnt/mini-sglang-cache2"
CACHE = modal.Volume.from_name(
    "mini-sglang-cache2",
)
MINI_REMOTE_ROOT = "/root/mini-sglang"

OUTPUT_DIR = Path("/private/tmp/mini_main_qwen3-8_test")
RAW_PATH = OUTPUT_DIR / "raw_tokens_logprobs.json"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"

DATASET_REPO = "abisee/cnn_dailymail"
DATASET_CONFIG = "3.0.0"
DATASET_REVISION = "96df5e686bee6baa90b8bee7c28b81fa3fa6223d"
MODEL = "Qwen/Qwen3-8B"
SEED = 0
NUM_CASES = 200
MAX_INPUT_TOKENS = 768
MAX_OUTPUT_TOKENS = 256
BATCH_SIZE = 1
NGRAM_SIZE = 3
NUM_DRAFT_TOKENS = 4

mini_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "libnuma1")
    .pip_install("uv", "datasets")
    .run_commands(
        "git clone --branch n_gram --depth 1 "
        "https://github.com/bluecoffee8/mini-sglang.git /root/mini-sglang",
        f"cd {MINI_REMOTE_ROOT} && uv venv --python=3.12",
        f"cd {MINI_REMOTE_ROOT} && . .venv/bin/activate && "
        "uv pip install -e . pytest 'datasets>=3,<5'",
    )
)

main_image = (
    modal.Image.from_registry("lmsysorg/sglang:dev-cu12")
    .pip_install("datasets", "transformers")
)


def _configure_cache_env(*, main_sglang: bool = False) -> None:
    os.environ["HF_HOME"] = f"{CACHE_ROOT}/huggingface"
    os.environ["XDG_CACHE_HOME"] = CACHE_ROOT
    os.environ["FLASHINFER_WORKSPACE_BASE"] = f"{CACHE_ROOT}/flashinfer"
    os.environ["TVM_FFI_CACHE_DIR"] = f"{CACHE_ROOT}/tvm-ffi"
    os.environ["TORCH_EXTENSIONS_DIR"] = f"{CACHE_ROOT}/torch_extensions"
    if main_sglang:
        # The SGLang dev image currently reports an idle pool-leak invariant
        # before NGRAM generation starts under this standalone Engine harness.
        # Disable only that checker; inference settings stay unchanged.
        os.environ["SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE"] = "0"


def _load_cases() -> list[dict[str, str | int]]:
    from datasets import load_dataset

    dataset = load_dataset(
        DATASET_REPO,
        DATASET_CONFIG,
        split="test",
        revision=DATASET_REVISION,
        cache_dir=f"{os.environ['HF_HOME']}/datasets",
    )
    indices = random.Random(SEED).sample(range(len(dataset)), NUM_CASES)
    cases = []
    for case_index, dataset_index in enumerate(indices):
        row = dataset[dataset_index]
        cases.append(
            {
                "case_index": case_index,
                "dataset_index": dataset_index,
                "id": row["id"],
                "article": row["article"],
            }
        )
    return cases


def _case_metadata(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "case_index": case["case_index"],
            "dataset_index": case["dataset_index"],
            "case_id": case["id"],
        }
        for case in cases
    ]


def _tokenize_articles(tokenizer: Any, articles: list[str]) -> list[list[int]]:
    prompt_prefix = (
        "<|im_start|>user\n"
        "Summarize the following news article in 3-4 sentences. "
        "Return only the summary, without analysis or extra headings.\n\n"
    )
    prompt_suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_ids = tokenizer.encode(prompt_prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(prompt_suffix, add_special_tokens=False)
    keep_article_tokens = MAX_INPUT_TOKENS - len(prefix_ids) - len(suffix_ids)
    if keep_article_tokens <= 0:
        raise ValueError("MAX_INPUT_TOKENS too small for chat prompt wrapper")
    tokenized = []
    for article in articles:
        article_ids = tokenizer.encode(article, add_special_tokens=False)
        tokenized.append(prefix_ids + article_ids[:keep_article_tokens] + suffix_ids)
    return tokenized


def _first_mismatch(a: list[int], b: list[int]) -> int | None:
    for i, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def _length_stats(lengths: list[int]) -> dict[str, int | float]:
    return {
        "min": min(lengths),
        "mean": sum(lengths) / len(lengths),
        "max": max(lengths),
        "hit_max_tokens": sum(length == MAX_OUTPUT_TOKENS for length in lengths),
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = math.ceil(percentile * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def _compact_mini_result(
    *,
    label: str,
    cases: list[dict[str, Any]],
    worker_result: dict[str, Any],
    commit: str,
    elapsed_s: float,
) -> dict[str, Any]:
    rows = []
    for case, token_ids, logprobs in zip(
        cases,
        worker_result["token_ids"],
        worker_result["logprobs"],
        strict=True,
    ):
        rows.append(
            {
                "case_index": case["case_index"],
                "dataset_index": case["dataset_index"],
                "case_id": case["id"],
                "output_ids": token_ids,
                "output_logprobs": logprobs,
                "output_len": len(token_ids),
            }
        )

    return {
        "engine": "mini-sglang",
        "label": label,
        "model": MODEL,
        "mini_commit": commit,
        "elapsed_s": elapsed_s,
        "settings": {
            "prompt_format": worker_result.get("prompt_format"),
            "attention_backend": "fi",
            "cache_type": "naive",
            "cuda_graph_max_bs": 0,
            "disable_overlap_scheduling_env": True,
            "page_size": 1,
            "max_running_req": BATCH_SIZE,
            "batch_size": BATCH_SIZE,
            "max_input_tokens": MAX_INPUT_TOKENS,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "ignore_eos": False,
            "ngram_size": NGRAM_SIZE if label == "speculative" else 0,
            "num_draft_tokens": NUM_DRAFT_TOKENS if label == "speculative" else 0,
        },
        "speculative_stats": worker_result.get("speculative_stats", {}),
        "rows": rows,
    }


def _extract_main_output_ids(output: dict[str, Any]) -> list[int]:
    for key in ("output_ids", "output_token_ids", "token_ids"):
        value = output.get(key)
        if isinstance(value, list):
            return [int(x) for x in value]
    meta = output.get("meta_info") or {}
    for key in ("output_ids", "output_token_ids", "token_ids"):
        value = meta.get(key)
        if isinstance(value, list):
            return [int(x) for x in value]
    token_logprobs = meta.get("output_token_logprobs")
    if isinstance(token_logprobs, list):
        token_ids = []
        for item in token_logprobs:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                token_ids.append(int(item[1]))
            elif isinstance(item, dict):
                for key in ("token_id", "id"):
                    if key in item:
                        token_ids.append(int(item[key]))
                        break
        if token_ids:
            return token_ids
    raise RuntimeError(
        "Could not extract output token ids from SGLang response. "
        f"Top-level keys={list(output.keys())}, meta keys={list(meta.keys())}"
    )


def _extract_main_output_logprobs(output: dict[str, Any]) -> list[float | None]:
    meta = output.get("meta_info") or {}
    token_logprobs = meta.get("output_token_logprobs")
    if not isinstance(token_logprobs, list):
        return []
    values: list[float | None] = []
    for item in token_logprobs:
        if isinstance(item, (list, tuple)) and item:
            values.append(None if item[0] is None else float(item[0]))
        elif isinstance(item, dict):
            value = item.get("logprob")
            values.append(None if value is None else float(value))
    return values


def _main_engine_kwargs(*, speculative: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model_path": MODEL,
        "attention_backend": "flashinfer",
        "page_size": 1,
        "disable_overlap_schedule": True,
        "disable_decode_cuda_graph": True,
        "disable_prefill_cuda_graph": True,
        "max_running_requests": BATCH_SIZE,
        "context_length": MAX_INPUT_TOKENS + MAX_OUTPUT_TOKENS + 16,
        "mem_fraction_static": 0.80,
        "log_level": "error",
    }
    if speculative:
        kwargs.update(
            {
                "speculative_algorithm": "NGRAM",
                "speculative_num_draft_tokens": NUM_DRAFT_TOKENS,
                "speculative_num_steps": NUM_DRAFT_TOKENS,
                "speculative_ngram_match_type": "BFS",
                "speculative_ngram_max_trie_depth": 18,
            }
        )
    return kwargs


def _compare_outputs(
    *,
    name: str,
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    mismatch_rows = []
    prefix_logprob_deltas: list[float] = []
    length_mismatches = 0
    for left_row, right_row in zip(left["rows"], right["rows"], strict=True):
        left_ids = left_row["output_ids"]
        right_ids = right_row["output_ids"]
        mismatch_pos = _first_mismatch(left_ids, right_ids)
        prefix_len = len(left_ids) if mismatch_pos is None else mismatch_pos
        length_mismatches += int(len(left_ids) != len(right_ids))

        for i in range(
            min(
                prefix_len,
                len(left_row["output_logprobs"]),
                len(right_row["output_logprobs"]),
            )
        ):
            left_lp = left_row["output_logprobs"][i]
            right_lp = right_row["output_logprobs"][i]
            if left_lp is not None and right_lp is not None:
                prefix_logprob_deltas.append(abs(float(left_lp) - float(right_lp)))

        if mismatch_pos is not None:
            mismatch_rows.append(
                {
                    "case_index": left_row["case_index"],
                    "dataset_index": left_row["dataset_index"],
                    "case_id": left_row["case_id"],
                    "position": mismatch_pos,
                    "left_token": (
                        left_ids[mismatch_pos]
                        if mismatch_pos < len(left_ids)
                        else None
                    ),
                    "right_token": (
                        right_ids[mismatch_pos]
                        if mismatch_pos < len(right_ids)
                        else None
                    ),
                    "left_len": len(left_ids),
                    "right_len": len(right_ids),
                    "left_logprob": (
                        left_row["output_logprobs"][mismatch_pos]
                        if mismatch_pos < len(left_row["output_logprobs"])
                        else None
                    ),
                    "right_logprob": (
                        right_row["output_logprobs"][mismatch_pos]
                        if mismatch_pos < len(right_row["output_logprobs"])
                        else None
                    ),
                }
            )

    return {
        "name": name,
        "left": f"{left['engine']}:{left['label']}",
        "right": f"{right['engine']}:{right['label']}",
        "cases": len(left["rows"]),
        "token_mismatch_cases": len(mismatch_rows),
        "length_mismatch_cases": length_mismatches,
        "left_output_length_stats": _length_stats(
            [row["output_len"] for row in left["rows"]]
        ),
        "right_output_length_stats": _length_stats(
            [row["output_len"] for row in right["rows"]]
        ),
        "prefix_logprob_compared": len(prefix_logprob_deltas),
        "prefix_logprob_max_abs_delta": (
            max(prefix_logprob_deltas) if prefix_logprob_deltas else None
        ),
        "prefix_logprob_mean_abs_delta": (
            sum(prefix_logprob_deltas) / len(prefix_logprob_deltas)
            if prefix_logprob_deltas
            else None
        ),
        "prefix_logprob_p99_abs_delta": _percentile(prefix_logprob_deltas, 0.99),
        "first_mismatches": mismatch_rows[:20],
    }


def _build_summary(raw: dict[str, Any]) -> dict[str, Any]:
    mini_base = raw["outputs"]["mini_baseline"]
    mini_spec = raw["outputs"]["mini_speculative"]
    main_base = raw["outputs"]["main_baseline"]
    main_spec = raw["outputs"]["main_speculative"]
    comparisons = {
        "mini_baseline_vs_speculative": _compare_outputs(
            name="mini baseline vs ngram", left=mini_base, right=mini_spec
        ),
        "main_baseline_vs_speculative": _compare_outputs(
            name="main baseline vs ngram", left=main_base, right=main_spec
        ),
        "mini_vs_main_baseline": _compare_outputs(
            name="mini vs main baseline", left=mini_base, right=main_base
        ),
        "mini_vs_main_speculative": _compare_outputs(
            name="mini vs main speculative", left=mini_spec, right=main_spec
        ),
    }
    return {
        "experiment": "qwen3_0_6b_cnn200_bs1_flashinfer_ngram_chat_template",
        "model": MODEL,
        "dataset": {
            "repo": DATASET_REPO,
            "config": DATASET_CONFIG,
            "revision": DATASET_REVISION,
            "split": "test",
            "seed": SEED,
            "num_cases": NUM_CASES,
        },
        "settings": {
            "prompt_format": "qwen_chat_template_enable_thinking_false",
            "max_input_tokens": MAX_INPUT_TOKENS,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "ignore_eos": False,
            "batch_size": BATCH_SIZE,
            "mini_attention_backend": "fi",
            "main_attention_backend": "flashinfer",
            "disable_overlap": True,
            "disable_decode_cuda_graph": True,
            "page_size": 1,
            "ngram_size_mini": NGRAM_SIZE,
            "num_draft_tokens": NUM_DRAFT_TOKENS,
        },
        "versions": {
            "mini_commit": mini_base.get("mini_commit"),
            "main_sglang_version": main_base.get("sglang_version"),
            "main_torch_version": main_base.get("torch_version"),
            "main_cuda_version": main_base.get("cuda_version"),
            "gpu": main_base.get("gpu") or mini_base.get("gpu"),
        },
        "comparisons": comparisons,
        "raw_path": str(RAW_PATH),
        "summary_path": str(SUMMARY_PATH),
    }


def _run_mini_mode(speculative: bool) -> str:
    _configure_cache_env()
    subprocess.run(
        ["git", "fetch", "--depth", "1", "origin", "codex/ngram-speculative"],
        cwd=MINI_REMOTE_ROOT,
        check=True,
    )
    subprocess.run(
        ["git", "switch", "--detach", "FETCH_HEAD"],
        cwd=MINI_REMOTE_ROOT,
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=MINI_REMOTE_ROOT,
        text=True,
    ).strip()
    print(f"mini {'speculative' if speculative else 'baseline'} commit {commit}")

    cases = _load_cases()
    tmp = Path("/tmp/mini_main_compare")
    tmp.mkdir(parents=True, exist_ok=True)
    cases_path = tmp / "cases.json"
    result_path = tmp / ("mini_speculative.json" if speculative else "mini_baseline.json")
    cases_path.write_text(json.dumps(cases))

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{MINI_REMOTE_ROOT}/.venv/bin:" + env["PATH"],
            "MINISGL_DISABLE_OVERLAP_SCHEDULING": "1",
            "MINISGL_ATTENTION_BACKEND": "fi",
            "MINISGL_CNN_MODEL": MODEL,
            "MINISGL_NGRAM_SIZE": str(NGRAM_SIZE),
            "MINISGL_NUM_DRAFT_TOKENS": str(NUM_DRAFT_TOKENS),
        }
    )
    command = [
        f"{MINI_REMOTE_ROOT}/.venv/bin/python",
        "tests/integration/test_ngram_speculative_numerics.py",
        "--worker",
        "speculative" if speculative else "baseline",
        "--cases",
        str(cases_path),
        "--result",
        str(result_path),
        "--max-input-tokens",
        str(MAX_INPUT_TOKENS),
        "--max-output-tokens",
        str(MAX_OUTPUT_TOKENS),
        "--batch-size",
        str(BATCH_SIZE),
        "--ignore-eos",
        "0",
    ]
    started = time.time()
    completed = subprocess.run(
        command,
        cwd=MINI_REMOTE_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"mini {'speculative' if speculative else 'baseline'} failed "
            f"with {completed.returncode}\nstdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    worker_result = json.loads(result_path.read_text())
    payload = _compact_mini_result(
        label="speculative" if speculative else "baseline",
        cases=cases,
        worker_result=worker_result,
        commit=commit,
        elapsed_s=time.time() - started,
    )
    payload["gpu"] = worker_result.get("gpu")
    return json.dumps(payload)


@APP.function(image=mini_image, gpu="H100!", timeout=2 * 60 * 60, volumes={CACHE_ROOT: CACHE})
def run_mini_baseline() -> str:
    return _run_mini_mode(False)


@APP.function(image=mini_image, gpu="H100!", timeout=2 * 60 * 60, volumes={CACHE_ROOT: CACHE})
def run_mini_speculative() -> str:
    return _run_mini_mode(True)


def _run_main_mode(speculative: bool) -> str:
    _configure_cache_env(main_sglang=True)

    import sglang as sgl
    import torch
    from transformers import AutoTokenizer

    cases = _load_cases()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompt_ids = _tokenize_articles(tokenizer, [str(c["article"]) for c in cases])

    kwargs = _main_engine_kwargs(speculative=speculative)
    llm = sgl.Engine(**kwargs)
    rows = []
    started = time.time()
    try:
        for case, prompt in zip(cases, prompt_ids, strict=True):
            output = llm.generate(
                input_ids=[prompt],
                sampling_params={
                    "temperature": 0.0,
                    "max_new_tokens": MAX_OUTPUT_TOKENS,
                    "ignore_eos": False,
                },
                return_logprob=True,
                top_logprobs_num=1,
            )
            if isinstance(output, list):
                output = output[0]
            output_ids = _extract_main_output_ids(output)
            output_logprobs = _extract_main_output_logprobs(output)
            if output_logprobs and len(output_logprobs) != len(output_ids):
                raise RuntimeError(
                    f"logprob/token length mismatch for case {case['case_index']}: "
                    f"{len(output_logprobs)} vs {len(output_ids)}"
                )
            rows.append(
                {
                    "case_index": case["case_index"],
                    "dataset_index": case["dataset_index"],
                    "case_id": case["id"],
                    "output_ids": output_ids,
                    "output_logprobs": output_logprobs,
                    "output_len": len(output_ids),
                }
            )
    finally:
        llm.shutdown()

    payload = {
        "engine": "main-sglang",
        "label": "speculative" if speculative else "baseline",
        "model": MODEL,
        "sglang_version": getattr(sgl, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "elapsed_s": time.time() - started,
        "settings": {
            **kwargs,
            "batch_size": BATCH_SIZE,
            "max_input_tokens": MAX_INPUT_TOKENS,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "ignore_eos": False,
            "prompt_format": "qwen_chat_template_enable_thinking_false",
            "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "0",
        },
        "speculative_stats": {},
        "rows": rows,
    }
    return json.dumps(payload)


@APP.function(image=main_image, gpu="H100!", timeout=2 * 60 * 60, volumes={CACHE_ROOT: CACHE})
def run_main_baseline() -> str:
    return _run_main_mode(False)


@APP.function(image=main_image, gpu="H100!", timeout=2 * 60 * 60, volumes={CACHE_ROOT: CACHE})
def run_main_speculative() -> str:
    return _run_main_mode(True)


@APP.local_entrypoint()
def main() -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("running mini baseline")
    mini_baseline = json.loads(run_mini_baseline.remote())
    print(f"mini baseline complete in {mini_baseline['elapsed_s']:.1f}s")

    # print("running mini speculative")
    # mini_speculative = json.loads(run_mini_speculative.remote())
    # print(f"mini speculative complete in {mini_speculative['elapsed_s']:.1f}s")

    # print("running main baseline")
    # main_baseline = json.loads(run_main_baseline.remote())
    # print(f"main baseline complete in {main_baseline['elapsed_s']:.1f}s")

    # print("running main speculative")
    # main_speculative = json.loads(run_main_speculative.remote())
    # print(f"main speculative complete in {main_speculative['elapsed_s']:.1f}s")

    raw = {
        "experiment": "qwen3_8b_test",
        "case_metadata": [
            {
                "case_index": row["case_index"],
                "dataset_index": row["dataset_index"],
                "case_id": row["case_id"],
            }
            for row in mini_baseline["rows"]
        ],
        "outputs": {
            "mini_baseline": mini_baseline,
            # "mini_speculative": mini_speculative,
            # "main_baseline": main_baseline,
            # "main_speculative": main_speculative,
        },
    }
    summary = _build_summary(raw)
    RAW_PATH.write_text(json.dumps(raw, indent=2))
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return json.dumps(summary)

# modal run minisgl_modal2.py