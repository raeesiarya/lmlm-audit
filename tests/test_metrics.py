import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/lmlm-audit"))

from metrics import (
    contains_match,
    is_unknown,
    normalize_answer,
    score_prediction,
    summarize_audit_metrics,
    summarize_results,
)


def test_normalize_answer() -> None:
    assert normalize_answer("Spice Girls!") == "spice girls"
    assert normalize_answer("$69.7 million") == "69.7 million"


def test_score_prediction_exact_match() -> None:
    scores = score_prediction("Richard Mthetwa", "Richard Mthetwa")
    assert scores["exact_match"] == 1.0
    assert scores["contains_match"] == 1.0
    assert scores["unknown"] == 0.0
    assert scores["precision"] == 1.0
    assert scores["recall"] == 1.0
    assert scores["f1"] == 1.0


def test_contains_match() -> None:
    assert contains_match("Spice Girls, a girl group", "Spice Girls") == 1.0
    assert contains_match("Dutch", "Jørgensen") == 0.0


def test_is_unknown() -> None:
    assert is_unknown("unknown") == 1.0
    assert is_unknown("") == 1.0
    assert is_unknown("I don't know") == 1.0
    assert is_unknown("Spice Girls") == 0.0


def test_score_prediction_partial_overlap() -> None:
    scores = score_prediction(
        "Sanskrit scholars, poets, musicians",
        "family of musicians",
    )
    assert scores["exact_match"] == 0.0
    assert scores["contains_match"] == 0.0
    assert scores["unknown"] == 0.0
    assert scores["precision"] == 0.25
    assert scores["recall"] == 1 / 3
    assert round(scores["f1"], 3) == 0.286


def test_summarize_results() -> None:
    summary = summarize_results(
        [
            {
                "ground_truth": "Spice Girls",
                "model_output": "Spice Girls",
            },
            {
                "ground_truth": "Bihar, India",
                "model_output": "1956",
            },
        ]
    )

    assert summary["count"] == 2
    assert summary["exact_match"] == 0.5
    assert summary["contains_match"] == 0.5
    assert summary["unknown_rate"] == 0.0
    assert summary["precision"] == 0.5
    assert summary["recall"] == 0.5
    assert summary["f1"] == 0.5


def test_summarize_results_unknown_rate() -> None:
    summary = summarize_results(
        [
            {
                "ground_truth": "Spice Girls",
                "model_output": "unknown",
            },
            {
                "ground_truth": "Bihar, India",
                "model_output": "",
            },
        ]
    )

    assert summary["count"] == 2
    assert summary["unknown_rate"] == 1.0


def test_summarize_audit_metrics() -> None:
    summary = summarize_audit_metrics(
        [
            {
                "fact_id": 1,
                "prompt": "What is Geri Halliwell famous for?",
                "ground_truth": "Spice Girls",
                "state": "DEL-ON",
                "model_output": "Spice Girls",
            },
            {
                "fact_id": 1,
                "prompt": "What is Geri Halliwell famous for?",
                "ground_truth": "Spice Girls",
                "state": "DEL-OFF",
                "model_output": "unknown",
            },
            {
                "fact_id": 2,
                "prompt": "What is Nozinja's birth name?",
                "ground_truth": "Richard Mthetwa",
                "state": "DEL-ON",
                "model_output": "Richard Mthetwa",
            },
            {
                "fact_id": 2,
                "prompt": "What is Nozinja's birth name?",
                "ground_truth": "Richard Mthetwa",
                "state": "DEL-OFF",
                "model_output": "Richard Mthetwa",
            },
        ]
    )

    assert summary["paired_count"] == 2
    assert summary["parametric_leakage"] == 0.5
    assert summary["retrieval_mediated_correctness"] == 0.5
