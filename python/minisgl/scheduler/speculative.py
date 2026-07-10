from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple

import torch

if TYPE_CHECKING:
    from minisgl.core import Batch, Req


@dataclass(frozen=True)
class SpeculativeConfig:
    """Server-wide n-gram speculative decoding settings.

    Speculation only ever applies to individual greedy-sampled requests; non-greedy
    requests in the same batch are always bypassed (extend_len stays 1 for them), so
    mixed greedy/sampled traffic is naturally supported without any special-casing.
    """

    enabled: bool
    num_draft_tokens: int  # k: max number of draft tokens proposed per round
    min_match_len: int  # n: trailing-context length used to look up a prior occurrence


class NgramIndex:
    """Per-request incremental "prompt lookup" index.

    Maps the trailing n-gram of a request's own token history (prompt + generated
    output) to the position where that n-gram last occurred earlier in the same
    history, so we can propose the tokens that followed it as a draft continuation.

    The index is updated incrementally (amortized O(1) per new token): each call only
    folds in n-grams ending at positions appended since the previous call.

    Correctness note: the n-gram ending at the very last position of the current
    history is deliberately *not* inserted into the table until the *next* call. If it
    were inserted immediately, a lookup for that same trailing window would always
    resolve to itself (the freshest occurrence), which would make every request
    trivially "self-match" and never find a genuine earlier occurrence.
    """

    def __init__(self, n: int) -> None:
        assert n > 0
        self.n = n
        self.table: Dict[Tuple[int, ...], int] = {}
        self.indexed_len = 0  # history length already folded into `table`

    def update_and_propose(self, token_ids: List[int], k: int) -> List[int]:
        n = self.n
        total = len(token_ids)

        # Index all windows ending strictly before the last position (see docstring).
        boundary = total - 1
        while self.indexed_len + n <= boundary:
            pos = self.indexed_len
            self.table[tuple(token_ids[pos : pos + n])] = pos
            self.indexed_len += 1

        if k <= 0 or total < n:
            return []

        query = tuple(token_ids[total - n :])
        match_start = self.table.get(query)
        if match_start is None:
            return []

        cont_start = match_start + n
        if cont_start >= total:
            return []
        return token_ids[cont_start : min(cont_start + k, total)]


def _propose_draft(req: Req, config: SpeculativeConfig) -> List[int]:
    if not req.sampling_params.is_greedy:
        return []
    k_eff = min(config.num_draft_tokens, req.remain_len)
    if k_eff <= 0:
        return []
    if req.ngram_index is None:
        req.ngram_index = NgramIndex(config.min_match_len)
    index: NgramIndex = req.ngram_index
    token_ids: List[int] = req.input_ids.tolist()
    return index.update_and_propose(token_ids, k_eff)


def prepare_speculative_batch(
    batch: Batch,
    config: SpeculativeConfig,
    token_pool: torch.Tensor,
    device: torch.device,
) -> None:
    """Attach draft tokens to greedy requests in a decode batch, before `_prepare_batch`.

    For each request this decides how many extra positions to score this round
    (`req.device_len` is extended accordingly, so KV-cache allocation and attention
    metadata prep transparently treat this as a ragged multi-token extend, exactly
    like chunked prefill) and writes the proposed draft tokens into the token pool so
    they are fed as teacher-forced input for the verify forward pass. Requests with no
    proposal (non-greedy, or no n-gram match found) get extend_len=1, i.e. an ordinary
    single-token decode step.
    """
    assert batch.is_decode
    write_idx: List[int] = []
    write_pos: List[int] = []
    write_val: List[int] = []
    for req in batch.reqs:
        draft = _propose_draft(req, config)
        req.pending_draft = draft
        req.device_len = req.cached_len + len(draft) + 1
        for offset, tok in enumerate(draft, start=1):
            write_idx.append(req.table_idx)
            write_pos.append(req.cached_len + offset)
            write_val.append(tok)

    if write_idx:
        idx_t = torch.tensor(write_idx, dtype=torch.int64, pin_memory=True).to(
            device, non_blocking=True
        )
        pos_t = torch.tensor(write_pos, dtype=torch.int64, pin_memory=True).to(
            device, non_blocking=True
        )
        val_t = torch.tensor(write_val, dtype=torch.int32, pin_memory=True).to(
            device, non_blocking=True
        )
        token_pool[idx_t, pos_t] = val_t


def compute_commit(
    draft: List[int],
    predicted: List[int],
    eos_token_id: int,
    ignore_eos: bool,
) -> List[int]:
    """Canonical (Leviathan et al.) greedy speculative-sampling acceptance rule.

    `predicted[i]` is the target model's exact argmax prediction for the token that
    follows draft position i (predicted[len(draft)] is the unconditional "bonus"
    token). Accept the longest prefix where predicted == draft, then take one more
    token (either the correction at the first mismatch, or the bonus token if the
    whole draft was accepted) -- this is deterministic and reproduces exactly what a
    plain non-speculative greedy decode would have produced token-by-token.
    """
    accept_len = 0
    while accept_len < len(draft) and predicted[accept_len] == draft[accept_len]:
        accept_len += 1
    committed = draft[:accept_len] + [predicted[accept_len]]
    if not ignore_eos and eos_token_id in committed:
        committed = committed[: committed.index(eos_token_id) + 1]
    return committed
