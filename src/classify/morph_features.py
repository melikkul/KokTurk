"""Morphological tag distribution features for text classification.

Extracts per-document statistics from atomized text for use as
features in sklearn pipelines (FeatureUnion with TF-IDF).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MorphTagFeatures(BaseEstimator, TransformerMixin):
    """Extract per-document morphological tag statistics as features.

    For each document, computes:
    - POS distribution (9 categories)
    - Case marker rates (7 cases, per 100 tokens)
    - Tense distribution (5 tenses)
    - Negation/plural ratios
    - Average suffix chain length
    - Lexical diversity (unique roots / total tokens)
    """

    POS_TAGS: list[str] = [
        "+Noun", "+Verb", "+Adj", "+Adv", "+Pron",
        "+Det", "+Postp", "+Conj", "+Num",
    ]
    CASE_TAGS: list[str] = [
        "+ACC", "+DAT", "+LOC", "+ABL", "+GEN", "+INS", "+NOM",
    ]
    TENSE_TAGS: list[str] = [
        "+PAST", "+FUT", "+AOR", "+PROG", "+EVID",
    ]

    def fit(
        self, X: list[str], y: object = None,  # noqa: N803
    ) -> MorphTagFeatures:
        """No-op fit (stateless transformer)."""
        return self

    def transform(self, X: list[str]) -> np.ndarray:  # noqa: N803
        """Extract morphological features from atomized documents.

        Args:
            X: List of atomized document strings. Each document is a
                space-separated sequence of "root +TAG1 +TAG2" tokens.

        Returns:
            Feature matrix of shape (n_docs, n_features).
        """
        features = []
        for doc in X:
            feat = self._extract_doc_features(doc)
            features.append(feat)
        return np.array(features, dtype=np.float64)

    def _extract_doc_features(self, doc: str) -> list[float]:
        """Extract features from a single document."""
        tokens = doc.split()
        n_tokens = max(len(tokens), 1)

        feat: list[float] = []

        # POS distribution (proportion of tokens containing each POS tag)
        for pos in self.POS_TAGS:
            feat.append(sum(1 for t in tokens if pos in t) / n_tokens)

        # Case marker rates (per 100 tokens)
        for case in self.CASE_TAGS:
            feat.append(sum(1 for t in tokens if case in t) / n_tokens * 100)

        # Tense distribution
        for tense in self.TENSE_TAGS:
            feat.append(sum(1 for t in tokens if tense in t) / n_tokens)

        # Negation ratio
        feat.append(sum(1 for t in tokens if "+NEG" in t) / n_tokens)

        # Plural ratio
        feat.append(sum(1 for t in tokens if "+PLU" in t) / n_tokens)

        # Average suffix chain length (count of + in each token)
        feat.append(
            np.mean([t.count("+") for t in tokens]).item() if tokens else 0.0,
        )

        # Lexical diversity: unique roots / total tokens
        # Root is the first space-separated part of each atomized token
        # In atomized text, tokens are separated by spaces within words
        # and words are separated by double spaces or newlines
        roots = []
        for t in tokens:
            if not t.startswith("+"):
                roots.append(t)
        n_roots = max(len(roots), 1)
        feat.append(len(set(roots)) / n_roots)

        return feat

    def get_feature_names_out(self, input_features: object = None) -> list[str]:
        """Return feature names for pipeline introspection."""
        names: list[str] = []
        for pos in self.POS_TAGS:
            names.append(f"pos_{pos.strip('+')}")
        for case in self.CASE_TAGS:
            names.append(f"case_{case.strip('+')}")
        for tense in self.TENSE_TAGS:
            names.append(f"tense_{tense.strip('+')}")
        names.extend(["neg_ratio", "plu_ratio", "avg_tags", "lexical_diversity"])
        return names
