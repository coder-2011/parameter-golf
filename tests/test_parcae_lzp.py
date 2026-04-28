import math
import sys
from pathlib import Path

import numpy as np

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


if __name__ == "__main__":
    for test_name, test_fn in list(globals().items()):
        if test_name.startswith("test_"):
            test_fn()
