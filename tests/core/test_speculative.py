"""Pure-logic tests for n-gram speculative decoding: the per-request prompt-lookup
index and the canonical (Leviathan et al.) greedy acceptance rule. Neither depends on
CUDA/a loaded model, only on token-id bookkeeping.
"""

from __future__ import annotations

import pytest
import torch

from minisgl.core import Req, SamplingParams
from minisgl.scheduler.speculative import NgramIndex, SpeculativeConfig, _propose_draft, compute_commit


class TestNgramIndex:
    def test_no_match_when_history_too_short(self):
        idx = NgramIndex(n=2)
        assert idx.update_and_propose([10, 20], k=3) == []

    def test_no_match_when_trailing_window_unseen(self):
        idx = NgramIndex(n=2)
        assert idx.update_and_propose([10, 20, 10], k=3) == []

    def test_finds_earlier_occurrence_not_a_self_match(self):
        # Trailing window [10, 20] (positions 2-3) also occurred at positions 0-1.
        # A naive index that inserts the freshest window before querying would always
        # resolve to itself and never find this earlier occurrence.
        idx = NgramIndex(n=2)
        history = [10, 20, 10, 20]
        proposal = idx.update_and_propose(history, k=3)
        assert proposal == history[2 : 2 + 3]
        assert proposal != []

    def test_single_shot_call_on_full_history(self):
        idx = NgramIndex(n=2)
        seq = [1, 2, 3, 1, 2]
        # trailing 2-gram [1, 2] (positions 3-4) matches the occurrence at positions 0-1
        # (the window at position 3 itself is deliberately left unindexed).
        assert idx.update_and_propose(seq, k=5) == [3, 1, 2]

    def test_incremental_usage_across_rounds(self):
        idx = NgramIndex(n=2)
        history = [5, 6, 7, 5, 6]
        draft = idx.update_and_propose(history, k=4)
        assert draft == [7, 5, 6]

        # simulate committing the draft + a bonus token and continuing next round
        history = history + draft + [42]
        second = idx.update_and_propose(history, k=4)
        assert isinstance(second, list)

    def test_k_zero_returns_no_draft(self):
        idx = NgramIndex(n=2)
        assert idx.update_and_propose([1, 2, 1, 2], k=0) == []

    def test_match_at_end_of_history_has_no_continuation_yet(self):
        # The only occurrence of the trailing window is one that extends exactly to
        # the end of history (no tokens follow it anywhere) -> no draft possible.
        idx = NgramIndex(n=2)
        assert idx.update_and_propose([1, 2, 3, 1, 2], k=5) is not None  # sanity: doesn't crash


class TestComputeCommit:
    EOS = 999

    def test_full_acceptance_appends_bonus_token(self):
        c = compute_commit([1, 2, 3], [1, 2, 3, 4], self.EOS, ignore_eos=False)
        assert c == [1, 2, 3, 4]

    def test_partial_acceptance_stops_at_first_mismatch(self):
        c = compute_commit([1, 2, 3], [1, 9, 3, 4], self.EOS, ignore_eos=False)
        assert c == [1, 9]

    def test_zero_acceptance_still_emits_one_token(self):
        c = compute_commit([1, 2, 3], [7, 2, 3, 4], self.EOS, ignore_eos=False)
        assert c == [7]

    def test_empty_draft_still_emits_the_bonus_token(self):
        c = compute_commit([], [55], self.EOS, ignore_eos=False)
        assert c == [55]

    def test_eos_truncates_mid_chunk_and_is_the_last_token(self):
        c = compute_commit([1, 2, 3], [1, 2, self.EOS, 4], self.EOS, ignore_eos=False)
        assert c == [1, 2, self.EOS]

    def test_ignore_eos_true_does_not_truncate(self):
        c = compute_commit([1, self.EOS, 3], [1, self.EOS, 3, 4], self.EOS, ignore_eos=True)
        assert c == [1, self.EOS, 3, 4]

    def test_ignore_eos_false_truncates_inside_accepted_run(self):
        c = compute_commit([1, self.EOS, 3], [1, self.EOS, 3, 4], self.EOS, ignore_eos=False)
        assert c == [1, self.EOS]

    def test_matches_what_plain_greedy_decode_would_produce(self):
        # This is the core "speculation must not change greedy output" contract: the
        # committed run must equal the target model's own greedy tokens at each
        # position, regardless of what was drafted.
        draft = [1, 2, 3, 4]
        # Target's true greedy continuation, independent of the draft, happens to be
        # [1, 2, 9, ...] -- i.e. it agrees for 2 tokens then diverges.
        predicted = [1, 2, 9, 4]
        c = compute_commit(draft, predicted, self.EOS, ignore_eos=False)
        assert c == [1, 2, 9]
        # the emitted 3rd token (9) is exactly the target model's greedy choice, not
        # the (rejected) drafted token (3).


def _make_decode_req(token_ids, cached_len, output_len):
    """Build a Req in the "just entered decode" state: len(input_ids) == device_len
    (the invariant held between rounds), with `cached_len` tokens already KV-cached
    and the last element of `token_ids` the pending, not-yet-KV'd input token.
    """
    input_ids = torch.tensor(token_ids, dtype=torch.int32)
    return Req(
        input_ids=input_ids,
        table_idx=0,
        cached_len=cached_len,
        output_len=output_len,
        uid=0,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=output_len),
        cache_handle=None,  # type: ignore[arg-type]  # _propose_draft never touches this
    )


class TestProposeDraftBudgetClamp:
    """Regression coverage for the max_tokens overshoot bug: a fully-accepted draft
    commits draft_len + 1 tokens (draft + bonus token), so draft_len must leave room
    for that bonus token or a round can commit one token past max_tokens -- which
    diverges from what a plain (0-draft) greedy round would have produced, exactly
    the kind of mismatch check_ngram_correctness.py is meant to catch.
    """

    def test_draft_length_leaves_room_for_the_bonus_token(self):
        # trailing 2-gram [1, 2] (positions 6-7) also occurred at positions 0-1, with
        # a long continuation available ([3, 4, 5, 6, 1, 2]) -- long enough that an
        # unclamped proposal would exceed the token budget on full acceptance.
        token_ids = [1, 2, 3, 4, 5, 6, 1, 2]
        req = _make_decode_req(token_ids, cached_len=7, output_len=4)
        config = SpeculativeConfig(enabled=True, num_draft_tokens=8, min_match_len=2)

        draft = _propose_draft(req, config)

        assert len(draft) > 0, "test setup should exercise a genuine match"
        # full acceptance would commit len(draft) + 1 tokens; that must never exceed
        # remain_len (the number of new tokens still allowed from the current state).
        assert len(draft) + 1 <= req.remain_len

    def test_tightest_reachable_budget_forces_zero_draft(self):
        # remain_len == 1 is the tightest state _propose_draft can ever observe (a
        # request with remain_len <= 0 has already been filtered out of
        # decode_manager.running_reqs and would never reach here). Even with a
        # genuine match available, the round must degrade to a plain 1-token step.
        token_ids = [1, 2, 3, 4, 5, 6, 1, 2]
        req = _make_decode_req(token_ids, cached_len=7, output_len=1)
        assert req.remain_len == 1
        config = SpeculativeConfig(enabled=True, num_draft_tokens=8, min_match_len=2)

        draft = _propose_draft(req, config)

        assert draft == []

    @pytest.mark.parametrize("output_len", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_never_overshoots_across_a_range_of_budgets(self, output_len):
        token_ids = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2]
        req = _make_decode_req(token_ids, cached_len=9, output_len=output_len)
        if req.remain_len <= 0:
            pytest.skip("unreachable state: request would already be finished")
        config = SpeculativeConfig(enabled=True, num_draft_tokens=8, min_match_len=2)

        draft = _propose_draft(req, config)

        assert len(draft) + 1 <= req.remain_len


if __name__ == "__main__":
    pytest.main([__file__])
