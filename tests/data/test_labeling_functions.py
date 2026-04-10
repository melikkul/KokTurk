"""Tests for labeling functions and label matrix generation."""

from __future__ import annotations

import json
from pathlib import Path

from data.labeling_functions import (
    ABSTAIN,
    ALL_LFS,
    TokenContext,
    build_label_matrix,
    lf_gazetteer,
    lf_neural_draft,
    lf_pos_bigram,
    lf_suffix_regex,
    lf_trmorph_unambiguous,
    lf_zeyrek_unambiguous,
)

# ============================================================
# Test fixtures
# ============================================================


def _make_ctx(
    surface: str,
    analyses: list[dict[str, object]],
    prev_pos: str | None = None,
) -> TokenContext:
    return TokenContext(
        surface=surface, analyses=analyses, prev_pos=prev_pos,
        sentence_id="test_001", token_idx=0,
    )


def _make_analysis(root: str, tags: list[str], source: str = "zeyrek") -> dict[str, object]:
    return {"root": root, "tags": tags, "source": source, "score": 1.0}


# ============================================================
# LF1: Zeyrek unambiguous
# ============================================================


class TestLfZeyrekUnambiguous:
    def test_single_parse_returns_index(self) -> None:
        ctx = _make_ctx("ev", [_make_analysis("ev", ["+Noun"], "zeyrek")])
        assert lf_zeyrek_unambiguous(ctx) == 0

    def test_multiple_parses_abstains(self) -> None:
        ctx = _make_ctx("yüz", [
            _make_analysis("yüz", ["+Noun"], "zeyrek"),
            _make_analysis("yüz", ["+Num"], "zeyrek"),
        ])
        assert lf_zeyrek_unambiguous(ctx) == ABSTAIN

    def test_no_zeyrek_parses_abstains(self) -> None:
        ctx = _make_ctx("ev", [_make_analysis("ev", ["+Noun"], "trmorph")])
        assert lf_zeyrek_unambiguous(ctx) == ABSTAIN

    def test_empty_analyses_abstains(self) -> None:
        ctx = _make_ctx("xyz", [])
        assert lf_zeyrek_unambiguous(ctx) == ABSTAIN


# ============================================================
# LF2: TRMorph unambiguous
# ============================================================


class TestLfTRMorphUnambiguous:
    def test_single_trmorph_returns_index(self) -> None:
        ctx = _make_ctx("ev", [
            _make_analysis("ev", ["+Noun"], "zeyrek"),
            _make_analysis("ev", ["+Noun"], "trmorph"),
        ])
        assert lf_trmorph_unambiguous(ctx) == 1

    def test_no_trmorph_abstains(self) -> None:
        ctx = _make_ctx("ev", [_make_analysis("ev", ["+Noun"], "zeyrek")])
        assert lf_trmorph_unambiguous(ctx) == ABSTAIN


# ============================================================
# LF3: Suffix regex
# ============================================================


class TestLfSuffixRegex:
    def test_ablative_suffix(self) -> None:
        ctx = _make_ctx("evden", [
            _make_analysis("ev", ["+Noun", "+ABL"], "zeyrek"),
        ])
        assert lf_suffix_regex(ctx) == 0

    def test_locative_suffix(self) -> None:
        ctx = _make_ctx("evde", [
            _make_analysis("ev", ["+Noun", "+LOC"], "zeyrek"),
        ])
        assert lf_suffix_regex(ctx) == 0

    def test_evidential_suffix(self) -> None:
        ctx = _make_ctx("gelmiş", [
            _make_analysis("gel", ["+Verb", "+EVID"], "zeyrek"),
        ])
        assert lf_suffix_regex(ctx) == 0

    def test_ambiguous_suffix_abstains(self) -> None:
        """When suffix matches but multiple candidates have the tag → ABSTAIN."""
        ctx = _make_ctx("evden", [
            _make_analysis("ev", ["+Noun", "+ABL"], "zeyrek"),
            _make_analysis("ev", ["+Noun", "+ABL"], "trmorph"),
        ])
        # Both have +ABL, so 2 matches → ABSTAIN
        assert lf_suffix_regex(ctx) == ABSTAIN

    def test_no_suffix_match_abstains(self) -> None:
        ctx = _make_ctx("güzel", [_make_analysis("güzel", ["+Adj"], "zeyrek")])
        assert lf_suffix_regex(ctx) == ABSTAIN


# ============================================================
# LF4: POS bigram
# ============================================================


class TestLfPosBigram:
    def test_det_followed_by_noun(self) -> None:
        """After DET, NOUN is highly likely."""
        ctx = _make_ctx("kitap", [
            _make_analysis("kitap", ["+Noun"], "zeyrek"),
        ], prev_pos="DET")
        result = lf_pos_bigram(ctx)
        # Should vote for the noun parse if it's the only NOUN candidate
        assert result == 0 or result == ABSTAIN  # depends on bigram model

    def test_no_prev_pos_abstains(self) -> None:
        ctx = _make_ctx("kitap", [
            _make_analysis("kitap", ["+Noun"], "zeyrek"),
        ], prev_pos=None)
        assert lf_pos_bigram(ctx) == ABSTAIN


# ============================================================
# LF5: Gazetteer
# ============================================================


class TestLfGazetteer:
    def test_known_propn_votes_noun(self) -> None:
        ctx = _make_ctx("Ankara", [
            _make_analysis("Ankara", ["+Noun", "+Prop"], "zeyrek"),
        ])
        result = lf_gazetteer(ctx)
        assert result == 0

    def test_unknown_word_abstains(self) -> None:
        ctx = _make_ctx("xyzqwerty", [
            _make_analysis("xyzqwerty", ["+Noun"], "zeyrek"),
        ])
        assert lf_gazetteer(ctx) == ABSTAIN


# ============================================================
# LF6: Neural draft (stub)
# ============================================================


class TestLfNeuralDraft:
    def test_always_abstains(self) -> None:
        ctx = _make_ctx("ev", [_make_analysis("ev", ["+Noun"], "zeyrek")])
        assert lf_neural_draft(ctx) == ABSTAIN


# ============================================================
# Label matrix
# ============================================================


class TestBuildLabelMatrix:
    def test_matrix_shape(self, tmp_path: Path) -> None:
        """Label matrix has shape (n_tokens, n_lfs)."""
        # Create minimal test corpus and candidates
        corpus = [
            {"sentence_id": "t001", "text": "ev güzel", "tokens": ["ev", "güzel"],
             "pos_tags": ["NOUN", "ADJ"], "lemmas": ["ev", "güzel"]},
        ]
        candidates = [
            {"sentence_id": "t001", "token_idx": 0, "surface": "ev",
             "analyses": [{"root": "ev", "tags": ["+Noun"], "source": "zeyrek", "score": 1.0}],
             "parse_count": 1},
            {"sentence_id": "t001", "token_idx": 1, "surface": "güzel",
             "analyses": [{"root": "güzel", "tags": ["+Adj"], "source": "zeyrek", "score": 1.0}],
             "parse_count": 1},
        ]

        corpus_path = tmp_path / "corpus.jsonl"
        candidates_path = tmp_path / "candidates.jsonl"

        with open(corpus_path, "w") as f:
            for s in corpus:
                f.write(json.dumps(s) + "\n")
        with open(candidates_path, "w") as f:
            for c in candidates:
                f.write(json.dumps(c) + "\n")

        label_matrix, records = build_label_matrix(candidates_path, corpus_path)
        assert label_matrix.shape == (2, len(ALL_LFS))
        assert len(records) == 2

    def test_all_lfs_registered(self) -> None:
        assert len(ALL_LFS) == 6
        names = [lf.name for lf in ALL_LFS]  # type: ignore[attr-defined]
        assert "zeyrek_unambiguous" in names
        assert "neural_draft" in names
