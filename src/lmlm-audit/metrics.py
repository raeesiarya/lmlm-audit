import re
from collections import Counter
from typing import Any


TOKEN_PATTERN = re.compile(r"\d+\.\d+|\w+(?:[-']\w+)*", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


def normalize_answer(text: str) -> str:
    return " ".join(tokenize(text))


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def precision_recall_f1(prediction: str, ground_truth: str) -> dict[str, float]:
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(ground_truth)

    if not pred_tokens and not gold_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    if not pred_tokens or not gold_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    overlap = sum((pred_counter & gold_counter).values())

    if overlap == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def score_prediction(prediction: str, ground_truth: str) -> dict[str, float]:
    overlap_scores = precision_recall_f1(prediction, ground_truth)
    return {
        "exact_match": exact_match(prediction, ground_truth),
        **overlap_scores,
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {
            "count": 0,
            "exact_match": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    totals = {
        "exact_match": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }

    for result in results:
        scores = score_prediction(result["model_output"], result["ground_truth"])
        for metric_name in totals:
            totals[metric_name] += scores[metric_name]

    count = len(results)
    return {
        "count": count,
        **{metric_name: value / count for metric_name, value in totals.items()},
    }
