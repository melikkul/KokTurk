"""Tests for MorphTagFeatures sklearn transformer."""

from __future__ import annotations

import numpy as np
import pytest

from classify.morph_features import MorphTagFeatures


@pytest.fixture()
def transformer() -> MorphTagFeatures:
    return MorphTagFeatures()


@pytest.fixture()
def docs() -> list[str]:
    return [
        "ev +Noun +PLU kitap +Noun +ACC",  # 2 nouns
        "git +Verb +PAST gel +Verb +FUT",   # 2 verbs
        "güzel +Adj büyük +Adj",            # 2 adjectives
    ]


class TestFitTransform:
    """Basic fit/transform contract tests."""

    def test_fit_returns_self(
        self, transformer: MorphTagFeatures, docs: list[str],
    ) -> None:
        result = transformer.fit(docs)
        assert result is transformer

    def test_transform_returns_ndarray(
        self, transformer: MorphTagFeatures, docs: list[str],
    ) -> None:
        result = transformer.fit_transform(docs)
        assert isinstance(result, np.ndarray)

    def test_output_shape(
        self, transformer: MorphTagFeatures, docs: list[str],
    ) -> None:
        result = transformer.fit_transform(docs)
        # 9 POS + 7 case + 5 tense + 4 other = 25
        assert result.shape == (3, 25)

    def test_output_dtype(
        self, transformer: MorphTagFeatures, docs: list[str],
    ) -> None:
        result = transformer.fit_transform(docs)
        assert result.dtype == np.float64


class TestFeatureValues:
    """Semantic correctness of extracted features."""

    def test_noun_doc_has_high_noun_ratio(
        self, transformer: MorphTagFeatures,
    ) -> None:
        """A document full of nouns should have high pos_Noun feature."""
        noun_doc = ["ev +Noun kitap +Noun masa +Noun"]
        features = transformer.transform(noun_doc)
        names = transformer.get_feature_names_out()
        noun_idx = names.index("pos_Noun")
        verb_idx = names.index("pos_Verb")
        assert features[0, noun_idx] > features[0, verb_idx]

    def test_plural_doc_has_high_plu_ratio(
        self, transformer: MorphTagFeatures,
    ) -> None:
        """Documents with +PLU tokens should have nonzero plu_ratio."""
        plu_doc = ["ev +PLU +Noun kitap +PLU +Noun"]
        features = transformer.transform(plu_doc)
        names = transformer.get_feature_names_out()
        plu_idx = names.index("plu_ratio")
        assert features[0, plu_idx] > 0.0

    def test_lexical_diversity_all_unique(
        self, transformer: MorphTagFeatures,
    ) -> None:
        """All unique roots should give lexical_diversity = 1.0."""
        doc = ["ev kitap masa araba"]  # 4 unique roots, no tags
        features = transformer.transform(doc)
        names = transformer.get_feature_names_out()
        div_idx = names.index("lexical_diversity")
        assert features[0, div_idx] == pytest.approx(1.0)

    def test_verb_doc_higher_verb_ratio(
        self, transformer: MorphTagFeatures, docs: list[str],
    ) -> None:
        """Second doc (verbs) should have higher verb ratio than first (nouns)."""
        features = transformer.transform(docs)
        names = transformer.get_feature_names_out()
        verb_idx = names.index("pos_Verb")
        assert features[1, verb_idx] > features[0, verb_idx]


class TestFeatureNames:
    """Feature name introspection."""

    def test_feature_names_count(
        self, transformer: MorphTagFeatures,
    ) -> None:
        names = transformer.get_feature_names_out()
        assert len(names) == 25

    def test_feature_names_are_strings(
        self, transformer: MorphTagFeatures,
    ) -> None:
        names = transformer.get_feature_names_out()
        assert all(isinstance(n, str) for n in names)

    def test_feature_names_include_expected(
        self, transformer: MorphTagFeatures,
    ) -> None:
        names = transformer.get_feature_names_out()
        assert "pos_Noun" in names
        assert "case_ACC" in names
        assert "tense_PAST" in names
        assert "plu_ratio" in names
        assert "lexical_diversity" in names
