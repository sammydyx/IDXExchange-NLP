"""Week 7 language-only real-estate search-query intent classification."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


INTENT_LABELS = ("browsing", "researching", "high_intent_inquiry")


class QueryIntentClassifier:
    """Classify a query from its wording alone, without behavioral data.

    The labels describe probable search intent, not a user's likelihood of
    purchasing a property.  Predictions should therefore be presented as a
    language-based routing signal, never as a buyer or conversion score.
    """

    def __init__(self, max_features: int = 1_500, random_state: int = 42):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=max_features,
            sublinear_tf=True,
        )
        self.model = LogisticRegression(max_iter=1_000, random_state=random_state)
        self.is_trained = False

    @staticmethod
    def _validated_texts(queries: list[str] | tuple[str, ...]) -> list[str]:
        texts = [str(query or "").strip() for query in queries]
        if not texts or any(not query for query in texts):
            raise ValueError("queries must contain only non-empty strings")
        return texts

    def train(self, queries: list[str] | tuple[str, ...], labels: list[str] | tuple[str, ...]) -> "QueryIntentClassifier":
        texts = self._validated_texts(queries)
        labels = [str(label).strip() for label in labels]
        if len(texts) != len(labels):
            raise ValueError("queries and labels must have the same length")
        unknown = sorted(set(labels) - set(INTENT_LABELS))
        if unknown:
            raise ValueError(f"unknown intent labels: {unknown}")
        if len(set(labels)) < 2:
            raise ValueError("training requires at least two intent labels")

        matrix = self.vectorizer.fit_transform(texts)
        self.model.fit(matrix, labels)
        self.is_trained = True
        return self

    def _require_trained(self) -> None:
        if not self.is_trained:
            raise RuntimeError("train the classifier before predicting")

    def predict_details(self, query: str, uncertainty_threshold: float = 0.60) -> dict[str, Any]:
        """Return the predicted intent, confidence, and all class probabilities."""
        self._require_trained()
        text = self._validated_texts([query])[0]
        probabilities = self.model.predict_proba(self.vectorizer.transform([text]))[0]
        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])
        return {
            "intent": str(self.model.classes_[best_index]),
            "confidence": confidence,
            "is_uncertain": confidence < uncertainty_threshold,
            "probabilities": {
                str(label): float(probability)
                for label, probability in zip(self.model.classes_, probabilities)
            },
        }

    def predict(self, query: str) -> tuple[str, float]:
        """Return ``(intent, confidence)`` for one non-empty query."""
        result = self.predict_details(query)
        return result["intent"], result["confidence"]

    def classify_with_parser(self, query: str, parser: Any) -> dict[str, Any]:
        """Combine Week 4 structured filters with Week 7 language intent."""
        details = self.predict_details(query)
        filters = parser.parse(query)
        return {"query": query, "filters": filters.as_dict(), **details}

    def save(self, path: str | Path) -> None:
        self._require_trained()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    @classmethod
    def load(cls, path: str | Path) -> "QueryIntentClassifier":
        classifier = joblib.load(path)
        if not isinstance(classifier, cls):
            raise TypeError("saved artifact is not a QueryIntentClassifier")
        return classifier
