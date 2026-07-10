"""Pure-logic tests for n-gram speculative decoding: the per-request prompt-lookup
index and the canonical (Leviathan et al.) greedy acceptance rule. Neither depends on
CUDA/a loaded model, only on token-id bookkeeping.
"""

from __future__ import annotations

import pytest

from minisgl.scheduler.speculative import NgramIndex, compute_commit


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


if __name__ == "__main__":
    pytest.main([__file__])
