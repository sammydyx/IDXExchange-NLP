from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.query_intent_classifier import QueryIntentClassifier
from scripts.query_parser import QueryParser
from scripts.train_query_intent_classifier import load_dataset


@pytest.fixture()
def training_rows():
    return [
        ("show me homes in Irvine", "browsing"),
        ("find condos in San Diego", "browsing"),
        ("browse homes with pools", "browsing"),
        ("list townhomes in Pasadena", "browsing"),
        ("which areas have lower taxes", "researching"),
        ("compare HOA fees in Irvine", "researching"),
        ("what should I know about San Diego schools", "researching"),
        ("how do property taxes compare", "researching"),
        ("schedule a tour this weekend", "high_intent_inquiry"),
        ("book an open house viewing", "high_intent_inquiry"),
        ("I am ready to make an offer", "high_intent_inquiry"),
        ("contact an agent today", "high_intent_inquiry"),
    ]


@pytest.fixture()
def classifier(training_rows):
    queries, labels = zip(*training_rows)
    return QueryIntentClassifier().train(list(queries), list(labels))


def test_predict_returns_known_label_and_probability(classifier):
    intent, confidence = classifier.predict("show homes in Irvine")
    assert intent in {"browsing", "researching", "high_intent_inquiry"}
    assert 0.0 <= confidence <= 1.0


def test_prediction_details_include_all_probabilities(classifier):
    details = classifier.predict_details("compare taxes in San Diego")
    assert set(details["probabilities"]) == {"browsing", "researching", "high_intent_inquiry"}
    assert sum(details["probabilities"].values()) == pytest.approx(1.0)
    assert isinstance(details["is_uncertain"], bool)


def test_integration_with_week4_parser(classifier):
    result = classifier.classify_with_parser("schedule a tour of 3 bedroom homes in Irvine under $1M", QueryParser())
    assert result["filters"]["city"] == "Irvine"
    assert result["filters"]["beds_min"] == 3
    assert result["filters"]["price_max"] == 1_000_000
    assert result["intent"] in {"browsing", "researching", "high_intent_inquiry"}


def test_predict_requires_training():
    with pytest.raises(RuntimeError, match="train"):
        QueryIntentClassifier().predict("show homes")


@pytest.mark.parametrize("query", ["", "   ", None])
def test_train_rejects_empty_query(query):
    with pytest.raises(ValueError, match="non-empty"):
        QueryIntentClassifier().train([query, "show homes"], ["browsing", "researching"])


def test_train_rejects_unknown_label():
    with pytest.raises(ValueError, match="unknown"):
        QueryIntentClassifier().train(["show homes", "find condos"], ["browsing", "unknown"])


def test_saved_model_round_trips(classifier, tmp_path):
    path = tmp_path / "classifier.joblib"
    classifier.save(path)
    loaded = QueryIntentClassifier.load(path)
    assert loaded.predict("show homes in Irvine") == classifier.predict("show homes in Irvine")


def test_week7_dataset_meets_size_and_label_requirements():
    root = Path(__file__).resolve().parents[1]
    dataframe = load_dataset(root / "data/processed/week7_intent_queries.csv")
    assert len(dataframe) >= 200
    assert dataframe["query"].nunique() == len(dataframe)
    assert set(dataframe["intent"]) == {"browsing", "researching", "high_intent_inquiry"}
    assert (dataframe["review_status"] == "manual_verified").all()


def test_dataset_has_balanced_manually_reviewed_labels():
    root = Path(__file__).resolve().parents[1]
    dataframe = pd.read_csv(root / "data/processed/week7_intent_queries.csv")
    assert dataframe["intent"].value_counts().to_dict() == {
        "browsing": 80,
        "researching": 80,
        "high_intent_inquiry": 80,
    }
