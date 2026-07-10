"""Correctness check for n-gram speculative decoding.

Per the canonical (Leviathan et al.) greedy speculative-sampling acceptance rule,
enabling speculation must never change what a *greedy* (temperature=0) request
generates -- an accepted draft token is only ever one the target model would have
picked anyway, and the "bonus"/correction token is always the target model's own
argmax. This script is the empirical, end-to-end check of that guarantee: it runs the
same greedy prompts through the offline engine with n-gram speculative decoding
disabled ("baseline") and enabled ("speculative"), and asserts the output token ids
are byte-for-byte identical.

Uses the offline `LLM` engine directly (see benchmark/offline/bench.py for the same
pattern) rather than going through the HTTP server, so token ids can be compared
exactly instead of round-tripping through detokenized text. Each engine attaches a
module-level global context that cannot be re-initialized within one process
(`minisgl.core._GLOBAL_CTX`), so a baseline run and a speculative run must be two
separate process invocations -- hence the run/compare split below.

Usage:
    python check_ngram_correctness.py run --mode baseline    --output /tmp/base.json
    python check_ngram_correctness.py run --mode speculative --output /tmp/spec.json
    python check_ngram_correctness.py compare /tmp/base.json /tmp/spec.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import List

from minisgl.core import SamplingParams
from minisgl.utils import init_logger

logger = init_logger(__name__)

MODEL = "Qwen/Qwen3-8B"


@dataclass(frozen=True)
class Case:
    name: str
    prompt: str
    max_tokens: int
    ignore_eos: bool


# A mix of prompts chosen to exercise different parts of the speculative-decoding
# path together, in one concurrently-batched request set:
#  - highly repetitive asks: frequent n-gram matches -> long accepted draft chains
#  - natural/unpredictable prose: rare/no matches -> falls back to plain 1-token decode
#  - a very small max_tokens budget: exercises the max-tokens-mid-draft truncation path
#  - ignore_eos=False with a prompt likely to emit EOS quickly: exercises the
#    EOS-mid-draft truncation path
CASES: List[Case] = [
    Case(
        name="repeat_phrase",
        prompt=(
            "Repeat the exact phrase 'the quick brown fox jumps over the lazy dog' "
            "30 times in a row, separated by spaces, and output nothing else."
        ),
        max_tokens=200,
        ignore_eos=True,
    ),
    Case(
        name="count_up",
        prompt=(
            "Count from 1 to 60. Write each number on its own line, formatted "
            "exactly as 'Number: X'."
        ),
        max_tokens=220,
        ignore_eos=True,
    ),
    Case(
        name="repeat_word",
        prompt="Write the word 'banana' 50 times, separated by commas, and nothing else.",
        max_tokens=180,
        ignore_eos=True,
    ),
    Case(
        name="natural_prose",
        prompt="In a few sentences, explain how photosynthesis works.",
        max_tokens=150,
        ignore_eos=True,
    ),
    Case(
        name="code_gen",
        prompt=(
            "Write a Python function `is_prime(n)` that checks primality, then call "
            "it on the numbers 2 through 12 and print each result."
        ),
        max_tokens=220,
        ignore_eos=True,
    ),
    Case(
        name="short_budget",
        prompt="Say hello in exactly one short sentence.",
        max_tokens=6,
        ignore_eos=True,
    ),
    Case(
        name="natural_eos",
        prompt="What is 2 + 2? Answer with only the number.",
        max_tokens=50,
        ignore_eos=False,
    ),
    Case(
        name="structured_list",
        prompt="List the first 20 prime numbers separated by commas, and nothing else.",
        max_tokens=160,
        ignore_eos=True,
    ),
]


def run(mode: str, model: str, output_path: str) -> None:
    assert mode in ("baseline", "speculative")
    from minisgl.llm import LLM

    kwargs = dict(
        # "auto" can resolve to a hybrid backend whose "fa" half is unusable on some
        # images (see minisgl_modal_bench.py); force flashinfer for both phases, same
        # as the throughput benchmark, so results are representative.
        attention_backend="fi",
    )
    if mode == "speculative":
        kwargs.update(
            speculative_algorithm="ngram",
            speculative_num_draft_tokens=8,
            speculative_ngram_min_match=3,
        )

    logger.info(f"Loading {model} in '{mode}' mode...")
    llm = LLM(model, **kwargs)
    try:
        llm.generate(["Warm up."], SamplingParams(temperature=0.0, max_tokens=4))

        prompts = [c.prompt for c in CASES]
        sampling_params = [
            SamplingParams(temperature=0.0, max_tokens=c.max_tokens, ignore_eos=c.ignore_eos)
            for c in CASES
        ]
        logger.info(f"Generating {len(CASES)} concurrently-batched cases...")
        start = time.time()
        results = llm.generate(prompts, sampling_params)
        elapsed = time.time() - start
    finally:
        llm.shutdown()

    payload = {
        "mode": mode,
        "model": model,
        "elapsed_s": elapsed,
        "cases": [
            {
                "name": c.name,
                "max_tokens": c.max_tokens,
                "ignore_eos": c.ignore_eos,
                "token_ids": r["token_ids"],
                "text": r["text"],
            }
            for c, r in zip(CASES, results, strict=True)
        ],
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Wrote {mode} results ({elapsed:.1f}s, {len(CASES)} cases) to {output_path}")


def compare(baseline_path: str, speculative_path: str) -> bool:
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(speculative_path) as f:
        speculative = json.load(f)

    base_cases = {c["name"]: c for c in baseline["cases"]}
    spec_cases = {c["name"]: c for c in speculative["cases"]}
    assert base_cases.keys() == spec_cases.keys(), (
        f"Case set mismatch: baseline={sorted(base_cases)} "
        f"speculative={sorted(spec_cases)}"
    )

    all_ok = True
    print(f"{'case':<16} {'baseline_len':>12} {'spec_len':>9}  status")
    print("-" * 60)
    for name in base_cases:
        b, s = base_cases[name], spec_cases[name]
        b_ids, s_ids = b["token_ids"], s["token_ids"]
        ok = b_ids == s_ids
        all_ok &= ok
        status = "OK" if ok else "MISMATCH"
        print(f"{name:<16} {len(b_ids):>12} {len(s_ids):>9}  {status}")
        if not ok:
            first_diff = next(
                (i for i, (x, y) in enumerate(zip(b_ids, s_ids)) if x != y),
                min(len(b_ids), len(s_ids)),
            )
            lo = max(0, first_diff - 3)
            print(f"    first divergence at output-token index {first_diff}:")
            print(f"    baseline    tokens: ...{b_ids[lo:first_diff + 5]}...")
            print(f"    speculative tokens: ...{s_ids[lo:first_diff + 5]}...")
            print(f"    baseline    text: {b['text'][:200]!r}")
            print(f"    speculative text: {s['text'][:200]!r}")

    print("-" * 60)
    print(f"baseline total time:    {baseline['elapsed_s']:.2f}s")
    if speculative["elapsed_s"] > 0:
        speedup = baseline["elapsed_s"] / speculative["elapsed_s"]
        print(f"speculative total time: {speculative['elapsed_s']:.2f}s  (speedup: {speedup:.2f}x)")
    print()
    if all_ok:
        print(
            "PASS: greedy output is byte-for-byte identical with and without "
            "n-gram speculative decoding."
        )
    else:
        print("FAIL: n-gram speculative decoding changed greedy output for at least one case.")
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL, help="HF model path/repo id.")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run one pass and save its output token ids.")
    run_p.add_argument("--mode", choices=["baseline", "speculative"], required=True)
    run_p.add_argument("--output", required=True, help="Path to write results JSON to.")

    cmp_p = sub.add_parser("compare", help="Compare a baseline and a speculative results file.")
    cmp_p.add_argument("baseline", help="Path to baseline results JSON.")
    cmp_p.add_argument("speculative", help="Path to speculative results JSON.")

    args = parser.parse_args()
    if args.command == "run":
        run(args.mode, args.model, args.output)
    else:
        ok = compare(args.baseline, args.speculative)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
