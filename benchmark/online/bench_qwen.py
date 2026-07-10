from __future__ import annotations

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


async def main():
    random.seed(42)  # reproducibility
    PORT = 1919
    N = 1000
    SCALES = [0.4, 0.5, 0.6, 0.7, 0.8, 1.6]  # from fast to slow
    async with OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="dummy") as client:
        MODEL = await get_model_name(client)
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        TRACES = read_qwen_trace(download_qwen_trace(URL), tokenizer, n=N, dummy=True)
        logger.info(f"Start benchmarking with {N} requests using model {MODEL}...")
        for scale in SCALES:
            traces = scale_traces(TRACES, scale)
            results = await benchmark_trace(client, traces, MODEL)
            process_benchmark_results(results)
        logger.info("Benchmarking completed.")


if __name__ == "__main__":
    asyncio.run(main())
