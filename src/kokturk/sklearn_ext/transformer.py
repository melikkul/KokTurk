"""scikit-learn integration for morphological atomization.

Usage:
    from kokturk.sklearn_ext import MorphoTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import SVC

    clf = Pipeline([
        ("morpho", MorphoTransformer(output="atomized")),
        ("tfidf", TfidfVectorizer()),
        ("svm", SVC()),
    ])
    clf.fit(train_texts, train_labels)
"""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore[import-untyped]

from kokturk.core.analyzer import MorphoAnalyzer


class MorphoTransformer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """scikit-learn transformer that atomizes Turkish text.

    Args:
        backends: Analyzer backends. Default: ["zeyrek"].
        output: Output mode — "roots", "atomized", or "tags".
            - "roots": space-separated roots only
            - "atomized": full atom strings ("ev +PLU +ABL")
            - "tags": only the tag sequences ("+PLU +ABL")
    """

    def __init__(
        self,
        backends: list[str] | None = None,
        output: str = "atomized",
    ) -> None:
        self.backends = backends or ["zeyrek"]
        self.output = output

    def fit(self, X: Any, y: Any = None) -> MorphoTransformer:  # noqa: N803
        """Fit — initializes the analyzer."""
        self.analyzer_ = MorphoAnalyzer(backends=self.backends)
        return self

    def transform(self, X: list[str]) -> list[str]:  # noqa: N803
        """Transform texts by atomizing each word.

        Args:
            X: List of text strings.

        Returns:
            List of transformed text strings.
        """
        results: list[str] = []
        for text in X:
            words = text.split()
            parts: list[str] = []
            for word in words:
                analysis = self.analyzer_.analyze(word)
                best = analysis.best
                if best is None:
                    parts.append(word)
                    continue
                if self.output == "roots":
                    parts.append(best.root)
                elif self.output == "atomized":
                    parts.append(best.to_str())
                elif self.output == "tags":
                    parts.append(" ".join(best.tags))
            results.append(" ".join(parts))
        return results
