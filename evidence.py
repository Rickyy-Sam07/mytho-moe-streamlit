"""Read the archived matched-QA evidence used in the paper."""

import csv
import io
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
EVIDENCE = ROOT / "Mytho_MoE_Datasets_and_Evidence_Candidate (2).zip"
MATCHED = "predictions/original_matched_scored_predictions.jsonl"
DOMAINS = {"hvm": "HVM", "rama": "RaMO"}
EXPERTS = {"hvm": "Harivamsha specialist", "rama": "Ramayana specialist"}
MODES = {
    "base": "Base Qwen3-4B",
    "mixed": "Mixed-Task Top-1",
    "adapted": "Graph-QA-Adapted Top-1",
}


@dataclass(frozen=True)
class Result:
    answer: str
    expert: str | None = None
    probabilities: dict[str, float] | None = None
    generation_likelihood: float | None = None
    generated_tokens: int | None = None
    generation_seconds: float | None = None
    router_seconds: float | None = None
    router_truncated: bool = False
    hit_generation_limit: bool = False


def load_evidence():
    bundled = APP_DIR / "matched-evidence.zip"
    with ZipFile(bundled if bundled.is_file() else EVIDENCE) as archive:
        with io.TextIOWrapper(archive.open(MATCHED), encoding="utf-8") as source:
            rows = [json.loads(line) for line in source if line.strip()]
        with io.TextIOWrapper(archive.open("results/summary.csv"), encoding="utf-8") as source:
            routing = {
                row["router"]: (float(row["routing_accuracy"]), int(row["unique_unambiguous_routing_queries"]))
                for row in csv.DictReader(source)
                if row["stratum"] == "all_provisional" and row["domain"] == "pooled"
                and row["variant"] == "original"
            }
    return rows, routing


def saved_result(row, mode):
    expert = row[f"{mode}_expert"] if mode != "base" else None
    output = row["answers"][expert or "base"]
    return Result(
        answer=output["answer"], expert=expert,
        probabilities=row[f"{mode}_probabilities"] if expert else None,
        generated_tokens=output["generated_tokens"], generation_seconds=output["generation_seconds"],
        hit_generation_limit=output["hit_generation_limit"],
    )


def score_answer(answer, references):
    def normalize(text):
        text = re.sub(r"\[Source:.*?\]", "", text, flags=re.I | re.S)
        return unicodedata.normalize("NFKC", text).casefold().split()

    prediction = normalize(answer)
    scores = []
    for reference in references:
        target = normalize(reference)
        shared = sum((Counter(prediction) & Counter(target)).values())
        precision = shared / len(prediction) if prediction else float(not target)
        recall = shared / len(target) if target else float(not prediction)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append((float(prediction == target), f1))
    return max(scores)


def benchmark(rows):
    """Pool the saved scores and measure each fixed expert on its own domain."""
    cohorts = {mode: rows for mode in MODES}
    cohorts.update({domain: [row for row in rows if row["domain"] == domain] for domain in DOMAINS})
    return {
        mode: {
            "n": len(cohort),
            **{metric: sum(row["scores"][mode][metric] for row in cohort) / len(cohort)
               for metric in ("exact_match", "token_f1")},
        }
        for mode, cohort in cohorts.items()
    }
