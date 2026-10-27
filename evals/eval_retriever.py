
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.models import OpenAIModel
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)

from rag_eval_project.reranker import RerankingRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

GOLDEN_PATH = PROJECT_ROOT / "goldens" / "retriever_goldens.json"

THRESHOLD = 0.7
FETCH_K = 20
TOP_K = 5

API_KEY = os.getenv("INCEPTION_API_KEY")
JUDGE_MODEL = os.getenv("INCEPTION_MODEL", "mercury-3")
BASE_URL = "https://api.inceptionlabs.ai/v1"


def main():
    if not API_KEY:
        raise ValueError("INCEPTION_API_KEY is missing from .env")

    with open(GOLDEN_PATH, encoding="utf-8") as f:
        goldens = json.load(f)

    if not goldens:
        raise ValueError("The golden dataset is empty.")

    retriever = RerankingRetriever(
        fetch_k=FETCH_K,
        top_k=TOP_K,
    )

    test_cases = []

    for g in goldens:
        retrieved = retriever.invoke(g["query"])

        retrieval_context = [
            doc.page_content for doc in retrieved
        ]

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                expected_output=g["ideal_answer"],
                retrieval_context=retrieval_context,
                actual_output="(generator not evaluated in this run)",
            )
        )

    judge = OpenAIModel(
        model=JUDGE_MODEL,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0,
    )

    metrics = [
        ContextualRecallMetric(
            threshold=THRESHOLD,
            model=judge,
            include_reason=True,
        ),
        ContextualPrecisionMetric(
            threshold=THRESHOLD,
            model=judge,
            include_reason=True,
        ),
    ]


    evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(
            run_async=True,
            max_concurrent=1,
            throttle_value=0,
        ),
        hyperparameters={
            "retriever": "reranked_fetch20_top7",
            "embedding_model": (
                "BAAI/bge-base-en-v1.5"
            ),
            "reranker": (
                "cross-encoder/"
                "ms-marco-MiniLM-L-6-v2"
            ),
            "fetch_k": FETCH_K,
            "top_k": TOP_K,
            "chunk_size": 1000,
            "chunk_overlap": 150,
            "judge_model": JUDGE_MODEL,
            "judge_provider": "Inception Labs",
            "golden_set": str(
                GOLDEN_PATH.relative_to(PROJECT_ROOT)
            ),
        },
    )


if __name__ == "__main__":
    main()
