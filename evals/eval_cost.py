"""
Operational eval: COST.

Measures actual Mercury 3 token usage and estimates LLM API cost.

Cost = uncached input tokens + cached input tokens + output tokens.

This evaluation measures LLM cost only. It excludes infrastructure,
embeddings, vector database, and reranking costs.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

from src.rag_pipeline import RagPipeline
from src.generator import prompt, llm

# Preserve AIMessage and its usage_metadata.
measured_chain = prompt | llm


# CONFIG

QUESTIONS = [
    "What is the difference between reference-based and reference-free evals?",
    "Explain what faithfulness measures in a RAG pipeline.",
    "How does the G-Eval metric assign a score?",
    "What is MMLU and why is contamination a problem?",
]

REPEATS = 3

# Mercury 3 pricing: USD per 1 million tokens.
PRICE_INPUT_PER_1M = 0.10
PRICE_OUTPUT_PER_1M = 0.35
PRICE_CACHED_INPUT_PER_1M = 0.01

# Business projection assumptions.
QUERIES_PER_DAY = 2000
USD_TO_INR = 88.0

# Example budget. This is a project-defined threshold, not a
# provider-defined limit.
COST_BUDGET_PER_QUERY_USD = 0.0015


# TOKEN MEASUREMENT

def measure_tokens(pipeline, question):
    """
    Retrieve real context and measure Mercury 3 token usage.
    """

    docs = pipeline.retriever.invoke(question)

    context_text = "\n\n".join(
        doc.page_content
        for doc in docs
        if doc.page_content and doc.page_content.strip()
    )

    msg = measured_chain.invoke({
        "question": question,
        "context": context_text,
    })

    usage = msg.usage_metadata

    if not usage:
        raise RuntimeError(
            "Mercury 3 did not return usage_metadata. "
            "Cannot calculate reliable token costs."
        )

    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")

    if input_tokens is None or output_tokens is None:
        raise RuntimeError(
            f"Incomplete token usage metadata: {usage}"
        )

    # Cached input tokens, if reported by the provider.
    details = usage.get("input_token_details") or {}

    cached_tokens = (
        details.get("cache_read")
        or details.get("cached_tokens")
        or 0
    )

    cached_tokens = min(cached_tokens, input_tokens)

    return {
        "input": input_tokens,
        "output": output_tokens,
        "cached": cached_tokens,
    }

# COST CALCULATION

def cost_usd(input_tokens, output_tokens, cached_tokens):
    """
    Calculate cost using separate cached and uncached input rates.
    """

    uncached_input = max(input_tokens - cached_tokens, 0)

    input_cost = (
        uncached_input / 1_000_000
    ) * PRICE_INPUT_PER_1M

    cached_cost = (
        cached_tokens / 1_000_000
    ) * PRICE_CACHED_INPUT_PER_1M

    output_cost = (
        output_tokens / 1_000_000
    ) * PRICE_OUTPUT_PER_1M

    total = input_cost + cached_cost + output_cost

    return {
        "input": input_cost,
        "cached": cached_cost,
        "output": output_cost,
        "total": total,
    }

# BENCHMARK

def benchmark(pipeline):
    rows = []

    print("Measuring Mercury 3 token usage...\n")

    for question in QUESTIONS:
        for repeat in range(REPEATS):

            tokens = measure_tokens(pipeline, question)

            costs = cost_usd(
                tokens["input"],
                tokens["output"],
                tokens["cached"],
            )

            rows.append({
                "question": question,
                **tokens,
                **{
                    f"cost_{key}": value
                    for key, value in costs.items()
                },
            })

            print(
                f"Question {QUESTIONS.index(question) + 1}, "
                f"Run {repeat + 1}: "
                f"Input={tokens['input']}, "
                f"Output={tokens['output']}, "
                f"Cached={tokens['cached']}, "
                f"Cost=${costs['total']:.8f}"
            )

    return rows

# AGGREGATION

def avg(rows, key):
    return sum(row[key] for row in rows) / len(rows)

# REPORT

def report(rows):

    n = len(rows)

    avg_input = avg(rows, "input")
    avg_output = avg(rows, "output")
    avg_cached = avg(rows, "cached")

    avg_cost = avg(rows, "cost_total")
    min_cost = min(row["cost_total"] for row in rows)
    max_cost = max(row["cost_total"] for row in rows)

    avg_input_cost = (
        avg(rows, "cost_input")
        + avg(rows, "cost_cached")
    )

    avg_output_cost = avg(rows, "cost_output")

    output_share = (
        100 * avg_output_cost / avg_cost
        if avg_cost else 0
    )

    print("\n" + "=" * 76)
    print("MERCURY 3 — RAG APPLICATION COST REPORT")
    print("=" * 76)

    print(f"Samples                 : {n}")
    print(f"Average input tokens    : {avg_input:.0f}")
    print(f"Average cached tokens   : {avg_cached:.0f}")
    print(f"Average output tokens   : {avg_output:.0f}")

    print("-" * 76)

    print(
        f"Average cost / query   : ${avg_cost:.8f} "
        f"(Rs {avg_cost * USD_TO_INR:.6f})"
    )

    print(
        f"Minimum / maximum      : "
        f"${min_cost:.8f} / ${max_cost:.8f}"
    )

    print(
        f"Input cost share       : "
        f"{100 - output_share:.1f}%"
    )

    print(
        f"Output cost share      : "
        f"{output_share:.1f}%"
    )

    print("-" * 76)

    # Business projection.
    daily = avg_cost * QUERIES_PER_DAY
    monthly = daily * 30

    print(f"Projection at {QUERIES_PER_DAY} queries/day:")

    print(
        f"Daily cost             : ${daily:.6f} "
        f"(Rs {daily * USD_TO_INR:.4f})"
    )

    print(
        f"Monthly cost           : ${monthly:.6f} "
        f"(Rs {monthly * USD_TO_INR:.4f})"
    )

    print("-" * 76)

    verdict = (
        "PASS"
        if avg_cost <= COST_BUDGET_PER_QUERY_USD
        else "FAIL"
    )

    print(
        f"Budget per query       : "
        f"${COST_BUDGET_PER_QUERY_USD:.6f}"
    )

    print(
        f"Average cost           : "
        f"${avg_cost:.8f} [{verdict}]"
    )

    print("=" * 76)

    print("\nPricing used (USD per 1M tokens):")
    print(f"Input                  : ${PRICE_INPUT_PER_1M}")
    print(f"Cached input           : ${PRICE_CACHED_INPUT_PER_1M}")
    print(f"Output                 : ${PRICE_OUTPUT_PER_1M}")

    print(
        "\nNote: Cached-token pricing is applied only when the API "
        "reports cached input tokens."
    )

    print(
        "This is an estimate based on the supplied rates and reported "
        "token usage. Actual billing may differ."
    )

# ENTRYPOINT

def main():

    pipeline = RagPipeline(
        fetch_k=20,
        top_k=5,
    )
    rows = benchmark(pipeline)
    report(rows)

if __name__ == "__main__":
    main()
