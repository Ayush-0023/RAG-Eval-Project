import json
import os
from pathlib import Path

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.models import OpenAIModel
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)

from rag_eval_project.retriever import build_retriever


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = PROJECT_ROOT / "goldens" / "retriever_goldens.json"

load_dotenv(PROJECT_ROOT / ".env")

# Evaluation configuration
THRESHOLD = 0.7
TOP_K = 5

# Inception Labs / Mercury 3
API_KEY = os.getenv("INCEPTION_API_KEY")
JUDGE_MODEL = os.getenv("INCEPTION_MODEL", "mercury-3")
BASE_URL = "https://api.inceptionlabs.ai/v1"


def main():
    # 1. Validate configuration
    if not API_KEY:
        raise ValueError(
            "INCEPTION_API_KEY is missing from the project-root .env file."
        )

    if not JUDGE_MODEL:
        raise ValueError(
            "INCEPTION_MODEL is missing from the project-root .env file."
        )

    if not GOLDEN_PATH.is_file():
        raise FileNotFoundError(
            f"Golden dataset not found: {GOLDEN_PATH}"
        )

    # 2. Load the golden dataset
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        goldens = json.load(f)

    if not isinstance(goldens, list) or not goldens:
        raise ValueError(
            "The golden dataset must be a non-empty JSON list."
        )

    for index, golden in enumerate(goldens):
        if not isinstance(golden, dict) or not all(
            key in golden for key in ("query", "ideal_answer")
        ):
            raise ValueError(
                f"Golden entry {index} must contain "
                "'query' and 'ideal_answer'."
            )

    print(f"Loaded {len(goldens)} golden test cases.")
    print(f"Judge model: {JUDGE_MODEL}")

    # 3. Retrieve documents for every golden question
    retriever = build_retriever()
    test_cases = []

    for golden in goldens:
        query = golden["query"]
        retrieved = retriever.invoke(query)

        retrieval_context = [
            doc.page_content for doc in retrieved
        ]

        print(
            f"Retrieved {len(retrieval_context)} chunks "
            f"for: {query}"
        )

        test_cases.append(
            LLMTestCase(
                input=query,
                expected_output=golden["ideal_answer"],
                retrieval_context=retrieval_context,
                actual_output=(
                    "(generator not evaluated in this run)"
                ),
            )
        )

    # 4. Configure Mercury 3 as the LLM judge
    judge = OpenAIModel(
        model=JUDGE_MODEL,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0,
    )

    # 5. Define retrieval metrics
    metrics = [
        ContextualRecallMetric(
            threshold=THRESHOLD,
            model=judge,
            include_reason=True,
            async_mode=False,
        ),
        ContextualPrecisionMetric(
            threshold=THRESHOLD,
            model=judge,
            include_reason=True,
            async_mode=False,
        ),
    ]

    # 6. Run the evaluation
    evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "retriever": "base_k5",
            "embedding_model": (
                "sentence-transformers/all-MiniLM-L6-v2"
            ),
            "chunk_size": 1000,
            "chunk_overlap": 150,
            "top_k": TOP_K,
            "judge_model": JUDGE_MODEL,
            "judge_provider": "Inception Labs",
            "golden_set": str(
                GOLDEN_PATH.relative_to(PROJECT_ROOT)
            ),
        },
    )


if __name__ == "__main__":
    main()