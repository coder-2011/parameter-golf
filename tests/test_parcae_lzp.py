import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train_gpt_parcae as pg


def _byte_token_luts():
    token_bytes = [bytes([i]) for i in range(256)]
    has_space = np.zeros(256, dtype=np.bool_)
    is_boundary = np.zeros(256, dtype=np.bool_)
    return token_bytes, has_space, is_boundary


def _ids_from_bytes(data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    prev = np.empty_like(target)
    prev[0] = -1
    prev[1:] = target[:-1]
    nll = np.full(target.shape, math.log(256.0), dtype=np.float64)
    return target, prev, nll


def test_parse_lzp_orders_deduplicates_and_prefers_longer_first():
    assert pg._parse_lzp_orders("4,8,4,6") == [8, 6, 4]


def test_lzp_context_match_uses_prediction_position_as_context_end():
    byte_stream = list(b"abcXabcY")

    assert pg._lzp_context_matches(byte_stream, pred_pos=3, cur_pos=7, order=3)
    assert not pg._lzp_context_matches(byte_stream, pred_pos=4, cur_pos=7, order=3)
    assert not pg._lzp_context_matches(byte_stream, pred_pos=7, cur_pos=7, order=3)


def test_lzp_rejects_hash_slot_collision_when_context_bytes_differ():
    token_bytes, has_space, is_boundary = _byte_token_luts()
    table_mask = 1
    first_context_key = (0 << 8) | 0
    colliding_context_key = (0 << 8) | 1
    assert pg._lzp_slot_from_context_key(first_context_key, table_mask) == pg._lzp_slot_from_context_key(
        colliding_context_key,
        table_mask,
    )
    assert pg._lzp_slot_from_context_key((0 << 8) | 2, table_mask) != pg._lzp_slot_from_context_key(
        first_context_key,
        table_mask,
    )
    assert pg._lzp_slot_from_context_key((2 << 8) | 0, table_mask) != pg._lzp_slot_from_context_key(
        first_context_key,
        table_mask,
    )
    target, prev, nll = _ids_from_bytes(bytes([0, 0, 2, 0, 1, 7]))

    result = pg._context_mixture_bpb(
        target,
        prev,
        nll,
        token_bytes,
        has_space,
        is_boundary,
        ppm_enabled=False,
        ppm_order=0,
        ppm_lambda_hi=1.0,
        ppm_lambda_lo=1.0,
        ppm_conf_threshold=1.0,
        ppm_escape_method="d",
        ppm_token_order=0,
        ppm_use_meta_mix=False,
        ppm_meta_alpha=0.995,
        ppm_meta_eta=2.0,
        ppm_meta_warmup_bytes=0,
        lzp_enabled=True,
        lzp_orders="2",
        lzp_table_bits=1,
        lzp_alpha_min=1.0,
        lzp_alpha_max=1.0,
        lzp_min_streak=0,
        lzp_max_streak=0,
        lzp_hit_prob=0.99,
    )

    assert result["lzp_coverage"] == 0.0
    assert math.isclose(result["lzp_only_bpb"], 8.0)


def test_lzp_improves_repeated_byte_stream_against_uniform_neural_baseline():
    token_bytes, has_space, is_boundary = _byte_token_luts()
    target, prev, nll = _ids_from_bytes((b"abcXYZ" * 12) + (b"tail" * 4))

    result = pg._context_mixture_bpb(
        target,
        prev,
        nll,
        token_bytes,
        has_space,
        is_boundary,
        ppm_enabled=False,
        ppm_order=0,
        ppm_lambda_hi=1.0,
        ppm_lambda_lo=1.0,
        ppm_conf_threshold=1.0,
        ppm_escape_method="d",
        ppm_token_order=0,
        ppm_use_meta_mix=False,
        ppm_meta_alpha=0.995,
        ppm_meta_eta=2.0,
        ppm_meta_warmup_bytes=0,
        lzp_enabled=True,
        lzp_orders="3,4,5",
        lzp_table_bits=10,
        lzp_alpha_min=0.20,
        lzp_alpha_max=0.80,
        lzp_min_streak=0,
        lzp_max_streak=3,
        lzp_hit_prob=0.99,
    )

    assert result["lzp_coverage"] > 0.5
    assert result["lzp_hit_rate"] > 0.8
    assert result["mix_bpb"] < result["nn_only_bpb"]


def test_context_mix_reports_ppm_and_lzp_metrics_together():
    token_bytes, has_space, is_boundary = _byte_token_luts()
    target, prev, nll = _ids_from_bytes(b"abracadabra " * 10)

    result = pg._context_mixture_bpb(
        target,
        prev,
        nll,
        token_bytes,
        has_space,
        is_boundary,
        ppm_enabled=True,
        ppm_order=4,
        ppm_lambda_hi=0.9,
        ppm_lambda_lo=0.05,
        ppm_conf_threshold=0.9,
        ppm_escape_method="d",
        ppm_token_order=0,
        ppm_use_meta_mix=False,
        ppm_meta_alpha=0.995,
        ppm_meta_eta=2.0,
        ppm_meta_warmup_bytes=0,
        lzp_enabled=True,
        lzp_orders="3,4",
        lzp_table_bits=10,
        lzp_alpha_min=0.05,
        lzp_alpha_max=0.30,
        lzp_min_streak=0,
        lzp_max_streak=4,
        lzp_hit_prob=0.98,
    )

    assert result["bytes"] == len(target)
    assert math.isfinite(result["mix_bpb"])
    assert math.isfinite(result["ppm_mix_bpb"])
    assert math.isfinite(result["ppm_only_bpb"])
    assert math.isfinite(result["lzp_only_bpb"])
    assert 0.0 <= result["lzp_coverage"] <= 1.0
    assert 0.0 <= result["lzp_hit_rate"] <= 1.0


def test_ppm_d_escape_improves_repeated_stream_and_logs_method():
    token_bytes, has_space, is_boundary = _byte_token_luts()
    target, prev, nll = _ids_from_bytes((b"abracadabra " * 30) + (b"banana " * 30))
    logs: list[str] = []

    result = pg._context_mixture_bpb(
        target,
        prev,
        nll,
        token_bytes,
        has_space,
        is_boundary,
        ppm_enabled=True,
        ppm_order=5,
        ppm_lambda_hi=0.9,
        ppm_lambda_lo=0.05,
        ppm_conf_threshold=0.9,
        ppm_escape_method="d",
        ppm_token_order=0,
        ppm_use_meta_mix=False,
        ppm_meta_alpha=0.995,
        ppm_meta_eta=2.0,
        ppm_meta_warmup_bytes=0,
        lzp_enabled=False,
        lzp_orders="3",
        lzp_table_bits=8,
        lzp_alpha_min=0.0,
        lzp_alpha_max=0.0,
        lzp_min_streak=0,
        lzp_max_streak=0,
        lzp_hit_prob=0.99,
        log_fn=logs.append,
    )

    assert result["cutoff"] == 0.0
    assert result["coverage"] == 1.0
    assert result["ppm_only_bpb"] < result["nn_only_bpb"]
    assert result["ppm_mix_bpb"] < result["nn_only_bpb"]
    assert any("ppm_escape:d" in line for line in logs)


def test_ppm_d_escape_beats_ppm_c_on_low_entropy_repeated_stream():
    token_bytes, has_space, is_boundary = _byte_token_luts()
    target, prev, nll = _ids_from_bytes(b"abcabcabcabc " * 80)

    def score(method: str) -> float:
        result = pg._context_mixture_bpb(
            target,
            prev,
            nll,
            token_bytes,
            has_space,
            is_boundary,
            ppm_enabled=True,
            ppm_order=6,
            ppm_lambda_hi=0.9,
            ppm_lambda_lo=0.05,
            ppm_conf_threshold=0.9,
            ppm_escape_method=method,
            ppm_token_order=0,
            ppm_use_meta_mix=False,
            ppm_meta_alpha=0.995,
            ppm_meta_eta=2.0,
            ppm_meta_warmup_bytes=0,
            lzp_enabled=False,
            lzp_orders="3",
            lzp_table_bits=8,
            lzp_alpha_min=0.0,
            lzp_alpha_max=0.0,
            lzp_min_streak=0,
            lzp_max_streak=0,
            lzp_hit_prob=0.99,
        )
        return result["ppm_only_bpb"]

    assert score("d") < score("c")


def test_context_mix_cutoff_reports_partial_coverage_and_finite_scores():
    token_bytes, has_space, is_boundary = _byte_token_luts()
    target, prev, nll = _ids_from_bytes(b"the quick brown fox jumps over the lazy dog " * 5000)

    result = pg._context_mixture_bpb(
        target,
        prev,
        nll,
        token_bytes,
        has_space,
        is_boundary,
        ppm_enabled=True,
        ppm_order=5,
        ppm_lambda_hi=0.9,
        ppm_lambda_lo=0.05,
        ppm_conf_threshold=0.9,
        ppm_escape_method="d",
        ppm_token_order=3,
        ppm_use_meta_mix=True,
        ppm_meta_alpha=0.995,
        ppm_meta_eta=2.0,
        ppm_meta_warmup_bytes=0,
        lzp_enabled=True,
        lzp_orders="3,4,5",
        lzp_table_bits=12,
        lzp_alpha_min=0.0,
        lzp_alpha_max=0.20,
        lzp_min_streak=0,
        lzp_max_streak=4,
        lzp_hit_prob=0.99,
        max_seconds=1e-9,
    )

    assert result["cutoff"] == 1.0
    assert result["coverage"] < 1.0
    assert result["scored_bytes"] < result["bytes"]
    assert math.isfinite(result["mix_bpb"])
    assert math.isfinite(result["ppm_mix_bpb"])


def test_lzp_disabled_is_neural_equivalent_even_with_repetition():
    token_bytes, has_space, is_boundary = _byte_token_luts()
    target, prev, nll = _ids_from_bytes(b"abcXYZ" * 20)

    result = pg._context_mixture_bpb(
        target,
        prev,
        nll,
        token_bytes,
        has_space,
        is_boundary,
        ppm_enabled=False,
        ppm_order=0,
        ppm_lambda_hi=1.0,
        ppm_lambda_lo=1.0,
        ppm_conf_threshold=1.0,
        ppm_escape_method="d",
        ppm_token_order=0,
        ppm_use_meta_mix=False,
        ppm_meta_alpha=0.995,
        ppm_meta_eta=2.0,
        ppm_meta_warmup_bytes=0,
        lzp_enabled=False,
        lzp_orders="3,4,5",
        lzp_table_bits=10,
        lzp_alpha_min=1.0,
        lzp_alpha_max=1.0,
        lzp_min_streak=0,
        lzp_max_streak=0,
        lzp_hit_prob=0.99,
    )

    assert result["lzp_coverage"] == 0.0
    assert result["lzp_hit_rate"] == 0.0
    assert math.isclose(result["lzp_only_bpb"], result["nn_only_bpb"])
    assert math.isclose(result["mix_bpb"], result["nn_only_bpb"])


def test_lzp_respects_multibyte_tokens_and_leading_space_bytes():
    token_bytes = [b"", b"ab", b"c"]
    has_space = np.array([False, True, False], dtype=np.bool_)
    is_boundary = np.array([True, False, False], dtype=np.bool_)
    target = np.array([1, 2, 1, 2, 1, 2, 1, 2], dtype=np.int64)
    prev = np.empty_like(target)
    prev[0] = 0
    prev[1:] = target[:-1]
    nll = np.full(target.shape, math.log(256.0), dtype=np.float64)

    byte_stream, byte_logp = pg._token_predictions_to_byte_stream(
        target,
        prev,
        nll,
        token_bytes,
        has_space,
        is_boundary,
    )
    assert bytes(byte_stream) == b"abc abc abc abc"
    assert len(byte_stream) == len(byte_logp)

    result = pg._context_mixture_bpb(
        target,
        prev,
        nll,
        token_bytes,
        has_space,
        is_boundary,
        ppm_enabled=False,
        ppm_order=0,
        ppm_lambda_hi=1.0,
        ppm_lambda_lo=1.0,
        ppm_conf_threshold=1.0,
        ppm_escape_method="d",
        ppm_token_order=0,
        ppm_use_meta_mix=False,
        ppm_meta_alpha=0.995,
        ppm_meta_eta=2.0,
        ppm_meta_warmup_bytes=0,
        lzp_enabled=True,
        lzp_orders="4",
        lzp_table_bits=8,
        lzp_alpha_min=1.0,
        lzp_alpha_max=1.0,
        lzp_min_streak=0,
        lzp_max_streak=0,
        lzp_hit_prob=0.99,
    )

    assert result["bytes"] == len(byte_stream)
    assert result["lzp_coverage"] > 0.25
    assert result["mix_bpb"] < result["nn_only_bpb"]


def test_lzp_long_order_uses_exact_context_check():
    token_bytes, has_space, is_boundary = _byte_token_luts()
    repeated = b"abcdefghijklZ" * 4
    target, prev, nll = _ids_from_bytes(repeated)

    result = pg._context_mixture_bpb(
        target,
        prev,
        nll,
        token_bytes,
        has_space,
        is_boundary,
        ppm_enabled=False,
        ppm_order=0,
        ppm_lambda_hi=1.0,
        ppm_lambda_lo=1.0,
        ppm_conf_threshold=1.0,
        ppm_escape_method="d",
        ppm_token_order=0,
        ppm_use_meta_mix=False,
        ppm_meta_alpha=0.995,
        ppm_meta_eta=2.0,
        ppm_meta_warmup_bytes=0,
        lzp_enabled=True,
        lzp_orders="12",
        lzp_table_bits=10,
        lzp_alpha_min=0.5,
        lzp_alpha_max=0.5,
        lzp_min_streak=0,
        lzp_max_streak=0,
        lzp_hit_prob=0.99,
    )

    assert result["lzp_coverage"] > 0.2
    assert result["lzp_hit_rate"] > 0.9
    assert result["mix_bpb"] < result["nn_only_bpb"]


def test_lzp_streak_gate_can_disable_mixing_without_disabling_diagnostics():
    token_bytes, has_space, is_boundary = _byte_token_luts()
    target, prev, nll = _ids_from_bytes(b"abcXYZ" * 16)

    result = pg._context_mixture_bpb(
        target,
        prev,
        nll,
        token_bytes,
        has_space,
        is_boundary,
        ppm_enabled=False,
        ppm_order=0,
        ppm_lambda_hi=1.0,
        ppm_lambda_lo=1.0,
        ppm_conf_threshold=1.0,
        ppm_escape_method="d",
        ppm_token_order=0,
        ppm_use_meta_mix=False,
        ppm_meta_alpha=0.995,
        ppm_meta_eta=2.0,
        ppm_meta_warmup_bytes=0,
        lzp_enabled=True,
        lzp_orders="3",
        lzp_table_bits=10,
        lzp_alpha_min=0.0,
        lzp_alpha_max=0.0,
        lzp_min_streak=99,
        lzp_max_streak=99,
        lzp_hit_prob=0.99,
    )

    assert result["lzp_coverage"] > 0.5
    assert result["lzp_hit_rate"] > 0.8
    assert result["lzp_avg_alpha"] == 0.0
    assert math.isclose(result["mix_bpb"], result["nn_only_bpb"])


def test_eval_val_sliding_lzp_integration_collects_prefix_once_and_logs_metrics():
    class UniformModel:
        def __init__(self, vocab_size: int):
            self.vocab_size = vocab_size
            self.training = True

        def eval(self):
            self.training = False

        def train(self):
            self.training = True

        def forward_logits(self, input_ids):
            shape = (*input_ids.shape, self.vocab_size)
            return torch.zeros(shape, dtype=torch.float32, device=input_ids.device)

    args = SimpleNamespace(
        train_seq_len=8,
        eval_stride=4,
        sliding_compile_logits=False,
        ppm_enabled=False,
        ppm_subset_tokens=0,
        ppm_order=0,
        ppm_lambda_hi=1.0,
        ppm_lambda_lo=1.0,
        ppm_conf_threshold=1.0,
        ppm_escape_method="d",
        ppm_token_order=0,
        ppm_use_meta_mix=False,
        ppm_meta_alpha=0.995,
        ppm_meta_eta=2.0,
        ppm_meta_warmup_bytes=0,
        context_mix_max_seconds=0.0,
        lzp_enabled=True,
        lzp_subset_tokens=24,
        lzp_orders_str="3,4",
        lzp_table_bits=8,
        lzp_alpha_min=0.25,
        lzp_alpha_max=0.50,
        lzp_min_streak=0,
        lzp_max_streak=2,
        lzp_hit_prob=0.99,
        val_byte_count_override=0,
    )
    data = (b"abcXYZ" * 6)
    val_tokens = torch.tensor(list(data), dtype=torch.int64)
    base_bytes_lut = torch.ones(256, dtype=torch.int16)
    has_space_lut = torch.zeros(256, dtype=torch.bool)
    boundary_lut = torch.zeros(256, dtype=torch.bool)
    token_bytes = [bytes([i]) for i in range(256)]
    logs: list[str] = []

    loss, bpb, context_result, ngram_result = pg.eval_val_sliding(
        args,
        UniformModel(256),
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_space_lut,
        is_boundary_token_lut=boundary_lut,
        token_bytes_lut=token_bytes,
        log_fn=logs.append,
        batch_seqs=2,
    )

    assert math.isclose(loss, math.log(256.0), rel_tol=1e-6)
    assert math.isclose(bpb, 8.0, rel_tol=1e-6)
    assert context_result is not None
    assert ngram_result is None
    assert context_result["bytes"] == args.lzp_subset_tokens
    assert context_result["lzp_coverage"] > 0.25
    assert any("context_mix bytes:24" in line and "lzp_only:" in line for line in logs)
    assert any("context_mix_time:" in line and "subset_tokens:24" in line for line in logs)


def test_eval_val_sliding_allows_sparse_stride_larger_than_window():
    class UniformModel:
        def __init__(self, vocab_size: int):
            self.vocab_size = vocab_size
            self.training = True

        def eval(self):
            self.training = False

        def train(self):
            self.training = True

        def forward_logits(self, input_ids):
            shape = (*input_ids.shape, self.vocab_size)
            return torch.zeros(shape, dtype=torch.float32, device=input_ids.device)

    args = SimpleNamespace(
        train_seq_len=8,
        eval_stride=16,
        sliding_compile_logits=False,
        ppm_enabled=False,
        ppm_subset_tokens=0,
        ppm_order=0,
        ppm_lambda_hi=1.0,
        ppm_lambda_lo=1.0,
        ppm_conf_threshold=1.0,
        ppm_escape_method="d",
        ppm_token_order=0,
        ppm_use_meta_mix=False,
        ppm_meta_alpha=0.995,
        ppm_meta_eta=2.0,
        ppm_meta_warmup_bytes=0,
        context_mix_max_seconds=0.0,
        lzp_enabled=False,
        lzp_subset_tokens=0,
        lzp_orders_str="3",
        lzp_table_bits=8,
        lzp_alpha_min=0.0,
        lzp_alpha_max=0.0,
        lzp_min_streak=0,
        lzp_max_streak=0,
        lzp_hit_prob=0.99,
        ngram_eval_order=0,
        val_byte_count_override=0,
    )
    val_tokens = torch.arange(40, dtype=torch.int64) % 16
    base_bytes_lut = torch.ones(256, dtype=torch.int16)
    has_space_lut = torch.zeros(256, dtype=torch.bool)
    boundary_lut = torch.zeros(256, dtype=torch.bool)
    token_bytes = [bytes([i]) for i in range(256)]

    context_size, chunk_windows = pg._ttt_chunk_windows(
        total_tokens=val_tokens.numel() - 1,
        seq_len=args.train_seq_len,
        stride=args.eval_stride,
        chunk_tokens=val_tokens.numel(),
    )

    assert context_size == 0
    assert chunk_windows == [[0, 16, 32]]

    loss, bpb, context_result, ngram_result = pg.eval_val_sliding(
        args,
        UniformModel(256),
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_space_lut,
        is_boundary_token_lut=boundary_lut,
        token_bytes_lut=token_bytes,
        batch_seqs=2,
    )

    assert math.isclose(loss, math.log(256.0), rel_tol=1e-6)
    assert math.isclose(bpb, 8.0, rel_tol=1e-6)
    assert context_result is None
    assert ngram_result is None


if __name__ == "__main__":
    for test_name, test_fn in list(globals().items()):
        if test_name.startswith("test_"):
            test_fn()
