
import os
import json

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.models import OpenAIModel
from deepeval.evaluate import AsyncConfig

from src.rag_pipeline import RagPipeline

load_dotenv()

GOLDEN_PATH = "goldens/correctness_goldens.json"
THRESHOLD = 0.7

JUDGE_MODEL = os.getenv("INCEPTION_MODEL", "mercury-3")


# MERCURY 3 JUDGE
class MercuryJudge(OpenAIModel):

    def supports_structured_outputs(self):
        return False

    def supports_json_mode(self):
        return True


judge = MercuryJudge(
    model=JUDGE_MODEL,
    api_key=os.getenv("INCEPTION_API_KEY"),
    base_url="https://api.inceptionlabs.ai/v1",
    temperature=0,
)

# LOAD GOLDEN DATASET
def load_goldens(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# APPLICATION QUALITY EVALUATION
def run(rag):

    # 1. LOAD GOLDEN QUESTIONS AND IDEAL ANSWERS

    goldens = load_goldens(GOLDEN_PATH)

    print(f"\nLoaded {len(goldens)} golden questions.")


    # 2. RUN THE COMPLETE RAG PIPELINE

    test_cases = []

    print("\nRunning RAG pipeline...\n")

    for i, g in enumerate(goldens, start=1):

        result = rag.invoke(g["question"])

        test_cases.append(
            LLMTestCase(
                input=g["question"],
                actual_output=result["answer"],
                expected_output=g["ideal_answer"],
            )
        )

        print(f"Completed query {i}/{len(goldens)}")


    print("\nPipeline execution completed.")


    # 3. CORRECTNESS
    correctness = GEval(
        name="Correctness",

        evaluation_steps=[
            "Compare the actual output against the key facts "
            "in the expected output.",

            "Heavily penalize statements in the actual output "
            "that contradict the expected output or are factually wrong.",

            "Reward statements that match the expected output "
            "in meaning, regardless of wording.",

            "Do NOT penalize the actual output for omitting information. "
            "Only wrong statements count here.",
        ],

        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],

        threshold=THRESHOLD,
        model=judge,
        strict_mode=False,
    )

    # 4. COMPLETENESS
    completeness = GEval(
        name="Completeness",

        evaluation_steps=[
            "Compare the actual output against the expected output "
            "and identify all the important facts and concepts "
            "that should be covered.",

            "Reward the actual output for covering all important "
            "points from the expected output.",

            "Penalize the actual output when it misses important "
            "information, explanations, or key concepts.",

            "Do not require identical wording. Evaluate whether "
            "the meaning and important information are covered.",

            "Do not penalize the answer for omitting minor details "
            "that are not essential to answering the question.",

            "Evaluate completeness independently of writing style "
            "or answer length.",
        ],

        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],

        threshold=THRESHOLD,
        model=judge,
        strict_mode=False,
    )


    # 5. STYLE
    style = GEval(
        name="Style",

        evaluation_steps=[
            "Evaluate whether the actual output is clear, readable, "
            "and easy for the user to understand.",

            "Reward answers that explain concepts in a logical "
            "and well-organized manner.",

            "Reward professional, respectful, and appropriate language.",

            "Penalize unnecessary repetition, confusing explanations, "
            "poor organization, and irrelevant information.",

            "Penalize excessive verbosity when it does not add "
            "useful information.",

            "Do not penalize an answer merely because it uses "
            "different wording or formatting from the expected output.",

            "Focus on presentation and communication quality, "
            "not factual correctness or completeness.",
        ],

        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],

        threshold=THRESHOLD,
        model=judge,
        strict_mode=False,
    )


    # 6. RUN DEEPEVAL
    print("\nStarting application quality evaluation...\n")

    evaluate(
        test_cases=test_cases,

        metrics=[
            correctness,
            completeness,
            style,
        ],

        async_config=AsyncConfig(
            run_async=True,
            max_concurrent=1,
            throttle_value=0,
        ),
    )

if __name__ == "__main__":
    rag = RagPipeline(
        fetch_k=20,
        top_k=5,
    )
    run(rag)