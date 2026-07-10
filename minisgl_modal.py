from __future__ import annotations

import json
import os

import modal


APP = modal.App("minisgl-ngram")
REMOTE_ROOT = "/root/mini-sglang"
CACHE_ROOT = "/mnt/mini-sglang-cache2"
CACHE = modal.Volume.from_name(
    "mini-sglang-cache2",
)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "libnuma1")
    .pip_install("uv", "datasets")
    .run_commands(
        "git clone --branch n_gram --depth 1 "
        "https://github.com/bluecoffee8/mini-sglang.git /root/mini-sglang",
        f"cd {REMOTE_ROOT} && uv venv --python=3.12",
        f"cd {REMOTE_ROOT} && . .venv/bin/activate && "
        "uv pip install -e . pytest 'datasets>=3,<5'",
    )
)


@APP.function(image=image, gpu="H100!", timeout=2 * 60 * 60, volumes={CACHE_ROOT: CACHE})
def smoke() -> str:
    os.environ["MINISGL_DISABLE_OVERLAP_SCHEDULING"] = "1"

    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    llm = LLM(
        "Qwen/Qwen3-8B",
        attention_backend="fa",
        cache_type="naive",
        cuda_graph_max_bs=0,
        max_extend_tokens=256,
        max_seq_len_override=2048,
        page_size=1,
        # speculative_ngram_size=1,
        # speculative_num_draft_tokens=4,
    )
    try:
        result = llm.generate(
            ["Repeat any short phrase you find useful, then explain why repetition is useful."],
            SamplingParams(ignore_eos=True, max_tokens=12),
        )
        stats = llm.speculator.stats
        assert stats.lookup_attempts > 0
        assert len(result) == 1 and len(result[0]["token_ids"]) == 12
        return json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "lookup_attempts": stats.lookup_attempts,
                "lookup_matches": stats.lookup_matches,
                "verify_steps": stats.verify_steps,
                "output_tokens": len(result[0]["token_ids"]),
            }
        )
    finally:
        llm.shutdown()

# modal run minisgl_modal.py