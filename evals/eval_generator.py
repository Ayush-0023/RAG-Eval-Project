"""
evals/eval_generator.py

Component-level evaluation of the generator in isolation.

Uses the golden context from faithfulness_dataset.json,
NOT the retriever's output.

Metrics:
- Faithfulness: Are generated claims supported by the context?
- Answer Relevancy: Does the answer address the question?
"""

import json
from pathlib import Path

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.models import OpenAIModel
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
)
from src.generator import generate


# Paths and environment
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

GOLDEN_PATH = PROJECT_ROOT / "goldens" / "faithfulness_dataset.json"

THRESHOLD = 0.7

API_KEY = __import__("os").getenv("INCEPTION_API_KEY")
JUDGE_MODEL = __import__("os").getenv(
    "INCEPTION_MODEL", "mercury-3"
)
BASE_URL = "https://api.inceptionlabs.ai/v1"


def main():
    if not API_KEY:
        raise ValueError("INCEPTION_API_KEY is missing from .env")

    # 1. Load golden dataset
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        goldens = json.load(f)

    if not goldens:
        raise ValueError("The faithfulness dataset is empty.")

    # 2. Configure Mercury 3 as the evaluation judge
    judge = OpenAIModel(
        model=JUDGE_MODEL,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0,
    )

    # 3. Generate answers using GOLDEN context only
    test_cases = []

    for index, g in enumerate(goldens):
        query = g["query"]
        context = g["ideal_context"]

        if not isinstance(context, list):
            raise ValueError(
                f"ideal_context must be a list for case {index}."
            )

        answer = generate(query, context)

        test_cases.append(
            LLMTestCase(
                input=query,
                actual_output=answer,
                retrieval_context=context,
            )
        )

        print(f"Generated answer for test case {index + 1}/{len(goldens)}")

    # 4. Define metrics
    metrics = [
        FaithfulnessMetric(
            threshold=THRESHOLD,
            model=judge,
            include_reason=True,
        ),
        AnswerRelevancyMetric(
            threshold=THRESHOLD,
            model=judge,
            include_reason=True,
        ),
    ]

    # 5. Evaluate
    evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(
            run_async=True,
            max_concurrent=1,
            throttle_value=0,
        ),
        hyperparameters={
            "component": "generator",
            "generator_model": JUDGE_MODEL,
            "judge_model": JUDGE_MODEL,
            "judge_provider": "Inception Labs",
            "golden_set": str(
                GOLDEN_PATH.relative_to(PROJECT_ROOT)
            ),
            "threshold": THRESHOLD,
        },
    )

if __name__ == "__main__":
    main()