"""Tests for classification pipeline components."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


class TestTfidfPipeline:
    def test_tfidf_logreg_basic(self) -> None:
        """TF-IDF + LogReg pipeline produces valid predictions."""
        texts = [
            "ev +Noun +PLU okul +Noun +DAT",
            "git +Verb +PAST gel +Verb +FUT",
            "ekonomi +Noun büyüme +Noun",
            "spor +Noun maç +Noun",
        ] * 10
        labels = [0, 1, 2, 3] * 10

        tfidf = TfidfVectorizer(ngram_range=(1, 2))
        x = tfidf.fit_transform(texts)
        clf = LogisticRegression(max_iter=200, random_state=42)
        clf.fit(x, labels)
        preds = clf.predict(x)

        assert len(preds) == len(labels)
        assert all(0 <= p <= 3 for p in preds)
        # Should overfit on training data
        assert f1_score(labels, preds, average="macro") > 0.5
