import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import download_hf_docs_and_tokenize as tok_export
import lossless_caps as lc


def test_lossless_caps_caseops_roundtrips_common_case_patterns():
    samples = [
        "The NASA Launch",
        "camelCase HTTPServer XMLHttpRequest",
        f"literal controls {lc.DEFAULT_V2_TITLE}{lc.DEFAULT_V2_ALLCAPS}{lc.DEFAULT_V2_CAPNEXT}{lc.DEFAULT_V2_ESC}",
        "emoji Cafe Éclair stays outside ASCII rules",
    ]

    for text in samples:
        encoded = lc.get_text_transform(lc.LOSSLESS_CAPS_CASEOPS_V1)(text)
        decoded = lc.get_text_inverse_transform(lc.LOSSLESS_CAPS_CASEOPS_V1)(encoded)
        assert decoded == text


def test_surface_piece_original_byte_counts_tracks_control_markers_across_pieces():
    surfaces = [
        lc.DEFAULT_V2_TITLE,
        "the",
        " ",
        lc.DEFAULT_V2_ALLCAPS,
        "na",
        "sa",
        " ",
        "x",
        lc.DEFAULT_V2_CAPNEXT,
        "m",
        "l",
    ]

    counts = lc.surface_piece_original_byte_counts(
        surfaces,
        text_transform_name=lc.LOSSLESS_CAPS_CASEOPS_V1,
    )

    assert counts == [0, 3, 1, 0, 2, 2, 1, 1, 0, 1, 1]
    assert sum(counts) == len("The NASA xMl")


def test_pure_byte_tokenizer_text_transform_option_records_manifest_and_encodes_transformed_text(tmp_path):
    spec = {
        "name": "byte_caps",
        "dataset_suffix": "byte_caps",
        "text_transform": lc.LOSSLESS_CAPS_CASEOPS_V1,
    }

    built = tok_export.build_pure_byte_tokenizer(
        spec=spec,
        docs_jsonl=tmp_path / "unused.jsonl",
        tokenizers_dir=tmp_path / "tokenizers",
    )

    encoded = built["encode"]("NASA")
    expected = np.frombuffer(lc.encode_lossless_caps_v2("NASA").encode("utf-8"), dtype=np.uint8).astype(np.uint16) + 4
    assert np.array_equal(encoded, expected)
    assert built["manifest"]["text_transform"] == lc.LOSSLESS_CAPS_CASEOPS_V1
    assert built["manifest"]["text_transform_control_symbols"] == [
        lc.DEFAULT_V2_TITLE,
        lc.DEFAULT_V2_ALLCAPS,
        lc.DEFAULT_V2_CAPNEXT,
        lc.DEFAULT_V2_ESC,
    ]


def test_sentencepiece_training_iterator_applies_text_transform(tmp_path):
    docs = tmp_path / "docs.jsonl"
    docs.write_text(
        json.dumps({"text": "The NASA Launch"}) + "\n",
        encoding="utf-8",
    )

    texts = list(
        tok_export._iter_sentencepiece_text(
            docs,
            text_transform=lc.LOSSLESS_CAPS_CASEOPS_V1,
        )
    )

    assert texts == [lc.encode_lossless_caps_v2("The NASA Launch")]
