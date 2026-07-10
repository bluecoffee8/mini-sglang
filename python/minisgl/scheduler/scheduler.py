from __future__ import annotations

from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from minisgl.core import Batch, Req
from minisgl.env import ENV
from minisgl.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from minisgl.utils import align_ceil, init_logger, load_tokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .speculative import SpeculativeConfig, compute_commit, prepare_speculative_batch
from .table import TableManager

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine

        self.engine = Engine(config)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens
        self.spec_config = SpeculativeConfig(
            enabled=config.speculative_algorithm != "none",
            num_draft_tokens=config.speculative_num_draft_tokens,
            min_match_len=config.speculative_ngram_min_match,
        )
        # self.config = config

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(last_data)
        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        # N-gram speculative decoding needs each round's actual accept length before
        # the next round can be scheduled correctly (KV-cache paging, running_reqs
        # membership, etc. all depend on it), which overlap scheduling can't provide
        # since it schedules round N+1 before round N's results are processed. Force
        # the synchronous loop whenever speculative decoding is enabled.
        if ENV.DISABLE_OVERLAP_SCHEDULING or self.spec_config.enabled:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    continue

                if req.pending_committed_tokens is not None:
                    # Speculative-decoding verify round: 1..k+1 tokens were already
                    # committed (cached_len/device_len finalized, append_host done) by
                    # _verify_speculative; here we only need to report them.
                    committed = req.pending_committed_tokens
                    req.pending_committed_tokens = None
                    finished = not req.can_decode
                    if not req.sampling_params.ignore_eos:
                        finished |= committed[-1] == self.eos_token_id
                    for j, tok in enumerate(committed):
                        is_last = j == len(committed) - 1
                        reply.append(
                            DetokenizeMsg(uid=req.uid, next_token=tok, finished=finished and is_last)
                        )
                else:
                    next_token = next_tokens_cpu[i]
                    req.append_host(next_token.unsqueeze(0))
                    next_token = int(next_token.item())
                    finished = not req.can_decode
                    if not req.sampling_params.ignore_eos:
                        finished |= next_token == self.eos_token_id
                    reply.append(
                        DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished)
                    )

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif batch.is_prefill:  # for prefill, non-chunk req, cache the prefix
                    self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs
        self.send_result(reply)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                return logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                self._free_req_resources(req_to_free)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _free_req_resources(self, req: Req) -> None:
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        self.engine.graph_runner.pad_batch(batch)
        self.cache_manager.allocate_paged(batch.reqs)
        batch.positions = _make_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]
        self.engine.attn_backend.prepare_metadata(batch)
        # Prefill requests always contribute exactly 1 logit row regardless of
        # extend_len (ParallelLMHead gathers the last position). Decode requests
        # contribute `extend_len` rows each; speculative-decoding verify rows
        # (extend_len > 1) are sampled separately via exact argmax in the engine, so
        # they're excluded here.
        sample_reqs = (
            batch.reqs if batch.is_prefill else [r for r in batch.reqs if r.extend_len == 1]
        )
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(sample_reqs),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        batch = self.prefill_manager.schedule_next_batch(self.prefill_budget)
        if batch is None:
            batch = self.decode_manager.schedule_next_batch()
            if batch is not None and self.spec_config.enabled:
                prepare_speculative_batch(batch, self.spec_config, self.token_pool, self.device)
        return self._prepare_batch(batch) if batch else None

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        forward_output = self.engine.forward_batch(batch, sample_args)
        if batch.has_speculative_rows:
            # Must finalize cached_len/device_len for speculating requests before
            # filter_reqs (right below) reads req.can_decode for the next round.
            self._verify_speculative(batch)
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output

    def _verify_speculative(self, batch: Batch) -> None:
        """Apply the greedy speculative-sampling acceptance rule to every request that
        scored >1 position this round, commit the accepted tokens (+ bonus/correction
        token), and reclaim the KV-cache pages that were spent on rejected positions.
        """
        spec_reqs = [r for r in batch.reqs if r.extend_len > 1]
        if not spec_reqs:
            return

        page_size = self.cache_manager.page_size
        page_table = self.table_manager.page_table
        write_idx: List[int] = []
        write_pos: List[int] = []
        write_val: List[int] = []
        free_chunks: List[torch.Tensor] = []

        for req in spec_reqs:
            assert req.spec_predicted is not None
            predicted: List[int] = req.spec_predicted.tolist()
            req.spec_predicted = None
            draft = req.pending_draft
            req.pending_draft = []

            committed = compute_commit(
                draft, predicted, self.eos_token_id, req.sampling_params.ignore_eos
            )
            speculative_device_len = req.device_len  # over-allocated extend, pre-verify
            req.finalize_step(len(committed))
            req.append_host(torch.tensor(committed, dtype=torch.int32))
            req.pending_committed_tokens = committed

            if req.can_decode:
                write_idx.append(req.table_idx)
                write_pos.append(req.cached_len)
                write_val.append(committed[-1])

            # Free the (page-aligned) tail of KV pages spent on positions beyond what
            # was actually committed. `free_from` rounds up to the first page fully
            # beyond the committed length, so a page straddling the accept boundary is
            # conservatively kept (harmless, bounded waste of <page_size tokens).
            # `free_to` must round UP (not down, and not left as the raw, possibly
            # unaligned `speculative_device_len`) to match the page-aligned extent
            # `allocate_paged` actually reserved -- `CacheManager.free_pages` extracts
            # one page-start per page_size-sized run of its input, so passing a
            # partial/misaligned page here would misidentify a page that's still
            # partly in use as fully free and hand it out to another request.
            free_from = align_ceil(req.cached_len, page_size)
            free_to = align_ceil(speculative_device_len, page_size)
            if free_from < free_to:
                free_chunks.append(page_table[req.table_idx, free_from:free_to])

        if write_idx:
            idx_t = torch.tensor(write_idx, dtype=torch.int64, pin_memory=True).to(
                self.device, non_blocking=True
            )
            pos_t = torch.tensor(write_pos, dtype=torch.int64, pin_memory=True).to(
                self.device, non_blocking=True
            )
            val_t = torch.tensor(write_val, dtype=torch.int32, pin_memory=True).to(
                self.device, non_blocking=True
            )
            self.token_pool[idx_t, pos_t] = val_t
        if free_chunks:
            self.cache_manager.free_pages(torch.cat(free_chunks))


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    # Speculative-decoding verify rows (decode phase, extend_len > 1) don't know their
    # real next-input write position until Scheduler._verify_speculative determines
    # the accept length, so route them to the same -1 scratch sentinel used for
    # finished requests; _verify_speculative writes their token itself afterward.
    write_list = [
        (req.device_len if (req.can_decode and not (batch.is_decode and req.extend_len > 1)) else -1)
        for req in batch.reqs
    ]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
