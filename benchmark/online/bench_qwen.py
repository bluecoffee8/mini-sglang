from __future__ import annotations

import argparse
import asyncio
import os
import random
from pathlib import Path

from minisgl.benchmark.client import (
    benchmark_trace,
    get_model_name,
    process_benchmark_results,
    read_qwen_trace,
    scale_traces,
)
from minisgl.utils import init_logger
from openai import AsyncOpenAI as OpenAI
from transformers import AutoTokenizer

logger = init_logger(__name__)

URL = "https://media.githubusercontent.com/media/alibaba-edu/qwen-bailian-usagetraces-anon/refs/heads/main/qwen_traceA_blksz_16.jsonl"


def download_qwen_trace(url: str, retries: int = 5) -> str:
    dir = Path(os.path.dirname(__file__))
    # download the file if not exists
    file_path = dir / "qwen_traceA_blksz_16.jsonl"
    if not file_path.exists():
        import urllib.error
        import urllib.request

        tmp_path = file_path.with_suffix(file_path.suffix + ".part")
        logger.info(f"Downloading trace from {url} to {file_path}...")
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                urllib.request.urlretrieve(url, tmp_path)
                tmp_path.rename(file_path)
                logger.info("Download completed.")
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
                last_error = e
                logger.warning(f"Download attempt {attempt}/{retries} failed: {e}")
                tmp_path.unlink(missing_ok=True)
        else:
            assert last_error is not None
            raise last_error
    return str(file_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a running minisgl server.")
    parser.add_argument("--port", type=int, default=1919, help="Port the server is listening on.")
    parser.add_argument("-n", "--num-requests", type=int, default=100, help="Number of requests.")
    parser.add_argument(
        "--speculative",
        action="store_true",
        help="Label this run as using n-gram speculative decoding in the log output. "
        "This does not itself enable speculative decoding -- that is a server-side "
        "setting (`--speculative-algorithm ngram`, see minisgl_modal_bench.py); this "
        "flag only affects how this run is labeled/printed for A/B comparison.",
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    random.seed(42)  # reproducibility
    PORT = args.port
    N = args.num_requests
    # SCALES = [0.4, 0.5, 0.6, 0.7, 0.8, 1.6]  # from fast to slow
    SCALES = [0.4, 1.6]
    label = "n-gram speculative decoding" if args.speculative else "baseline (no speculation)"
    async with OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="dummy") as client:
        MODEL = await get_model_name(client)
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        TRACES = read_qwen_trace(download_qwen_trace(URL), tokenizer, n=N, dummy=True)
        logger.info(f"=== Benchmark mode: {label} ===")
        logger.info(f"Start benchmarking with {N} requests using model {MODEL}...")
        for scale in SCALES:
            traces = scale_traces(TRACES, scale)
            results = await benchmark_trace(client, traces, MODEL)
            logger.info(f"--- Results for scale={scale} ({label}) ---")
            process_benchmark_results(results)
        logger.info(f"Benchmarking completed ({label}).")


if __name__ == "__main__":
    asyncio.run(main())
