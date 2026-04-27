from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import sentencepiece as spm

import eval_fineweb_bpb as bpb


def write_shard(path: Path, tokens: np.ndarray) -> None:
    header = np.zeros((bpb.SHARD_HEADER_INTS,), dtype="<i4")
    header[0] = bpb.SHARD_MAGIC
    header[1] = bpb.SHARD_VERSION
    header[2] = int(tokens.size)
    with path.open("wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype("<u2", copy=False).tobytes())


class EvalFineWebBpbTest(unittest.TestCase):
    def test_bpb_from_sums_is_total_bits_over_total_bytes(self) -> None:
        loss_sum = 123.5
        token_count = 100
        byte_count = 80
        val_loss, val_bpb = bpb.bpb_from_sums(loss_sum, token_count, byte_count)

        self.assertAlmostEqual(val_loss, loss_sum / token_count)
        self.assertAlmostEqual(val_bpb, loss_sum / (math.log(2.0) * byte_count))

    def test_token_byte_sum_counts_sentencepiece_leading_spaces(self) -> None:
        table_size = 1892
        base_bytes = np.zeros((table_size,), dtype=np.int16)
        has_leading_space = np.zeros((table_size,), dtype=np.bool_)
        is_boundary_token = np.ones((table_size,), dtype=np.bool_)
        base_bytes[10] = 3
        base_bytes[11] = 4
        has_leading_space[11] = True
        is_boundary_token[10] = False
        is_boundary_token[11] = False
        luts = bpb.TokenizerLuts(base_bytes, has_leading_space, is_boundary_token)

        prev_ids = np.array([0, 10, 11], dtype=np.uint16)
        tgt_ids = np.array([11, 11, 10], dtype=np.uint16)

        self.assertEqual(bpb.token_byte_sum(prev_ids, tgt_ids, luts), 4 + 5 + 3)

    def test_sentencepiece_luts_match_checked_in_model_decode_bytes(self) -> None:
        tokenizer = Path(__file__).resolve().parents[2] / "data/tokenizers/fineweb_1024_bpe.model"
        if not tokenizer.exists():
            self.skipTest(f"missing tokenizer fixture: {tokenizer}")
        sp = spm.SentencePieceProcessor(model_file=str(tokenizer))
        luts = bpb.build_sentencepiece_luts(tokenizer, 1024)

        for text in ["Hello world", "the quick brown fox", "cafe\nau lait"]:
            ids = np.array([sp.bos_id(), *sp.encode(text, out_type=int)], dtype=np.uint16)
            expected_bytes = len(sp.decode(ids.tolist()).encode("utf-8"))
            self.assertEqual(bpb.token_byte_sum(ids[:-1], ids[1:], luts), expected_bytes)

    def test_iter_score_spans_nonoverlap_scores_once(self) -> None:
        spans = bpb.iter_score_spans(total_targets=16, ctx_len=4, stride=4)

        self.assertEqual([(s.score_start, s.score_end) for s in spans], [(0, 4), (4, 8), (8, 12), (12, 16)])
        self.assertEqual([(s.window_start, s.window_end, s.rel_start, s.rel_end) for s in spans], [(0, 4, 0, 4), (4, 8, 0, 4), (8, 12, 0, 4), (12, 16, 0, 4)])

    def test_iter_score_spans_sliding_scores_each_target_once(self) -> None:
        spans = bpb.iter_score_spans(total_targets=16, ctx_len=8, stride=3)
        covered: list[int] = []
        for span in spans:
            covered.extend(range(span.score_start, span.score_end))
            self.assertLessEqual(span.window_end - span.window_start, 8)
            self.assertEqual(span.rel_end - span.rel_start, span.score_end - span.score_start)

        self.assertEqual(covered, list(range(16)))

    def test_load_u16_tokens_reads_fineweb_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard = root / "fineweb_val_000000.bin"
            tokens = np.array([1, 2, 3, 4, 5], dtype=np.uint16)
            write_shard(shard, tokens)

            files = bpb.resolve_val_files(root)
            loaded = bpb.load_u16_tokens(files)

        self.assertEqual(files, [shard])
        np.testing.assert_array_equal(loaded, tokens)


if __name__ == "__main__":
    unittest.main()
