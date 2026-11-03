"""
Operational Evaluation: LATENCY

Measures:
1. End-to-end latency
2. Time to first token (TTFT)
3. Retrieval latency
4. Generation latency

No golden dataset or LLM judge is required.
"""

import math
import time

from dotenv import load_dotenv

from src.rag_pipeline import RagPipeline
from src.generator import generate_stream

load_dotenv()

QUESTIONS = [
    "What is the difference between reference-based and reference-free evals?",
    "Explain what faithfulness measures in a RAG pipeline.",
    "How does the G-Eval metric assign a score?",
    "What is MMLU and why is contamination a problem?",
]

REPEATS = 5
WARMUP_RUNS = 2

MEASURE_TTFT = True
STAGE_LEVEL = True

# Latency budgets in milliseconds
SLO_P95_MS = 3000
SLO_TTFT_P95_MS = 1200

# PIPELINE ADAPTERS

def run_end_to_end(pipeline, question):

    result = pipeline.invoke(question)

    return result["answer"]

def run_stages_streaming(pipeline, question):

    # Start timing before retrieval
    t0 = time.perf_counter()

    # Retrieval + reranking
    docs = pipeline.retriever.invoke(question)

    context = [
        doc.page_content
        for doc in docs
    ]

    t1 = time.perf_counter()

    # Stream generation
    first_token_t = None
    pieces = []

    for piece in generate_stream(question, context):

        if piece:
            if first_token_t is None:
                first_token_t = time.perf_counter()

            pieces.append(piece)

    t2 = time.perf_counter()

    answer = "".join(pieces)

    ttft_ms = (
        (first_token_t - t0) * 1000
        if first_token_t is not None
        else float("nan")
    )

    return answer, {
        "retrieval": (t1 - t0) * 1000,
        "generation": (t2 - t1) * 1000,
        "ttft": ttft_ms,
    }


# PERCENTILE CALCULATION

def percentile(values, p):

    values = [
        v for v in values
        if not math.isnan(v)
    ]

    if not values:
        return float("nan")

    s = sorted(values)

    k = (len(s) - 1) * (p / 100.0)

    lo = math.floor(k)
    hi = math.ceil(k)

    if lo == hi:
        return s[int(k)]

    return (
        s[lo] * (hi - k)
        + s[hi] * (k - lo)
    )


# BENCHMARK

def benchmark(pipeline):

    print(
        f"\nWarming up ({WARMUP_RUNS} runs, discarded)..."
    )

    for i in range(WARMUP_RUNS):

        run_end_to_end(
            pipeline,
            QUESTIONS[i % len(QUESTIONS)]
        )

    total_ms = []
    retrieval_ms = []
    generation_ms = []
    ttft_ms = []
    answer_lengths = []

    print("\nMeasuring latency...\n")

    for question in QUESTIONS:

        for _ in range(REPEATS):

            start = time.perf_counter()

            if MEASURE_TTFT:

                answer, stage = run_stages_streaming(
                    pipeline,
                    question
                )

                retrieval_ms.append(
                    stage["retrieval"]
                )

                generation_ms.append(
                    stage["generation"]
                )

                ttft_ms.append(
                    stage["ttft"]
                )

            elif STAGE_LEVEL:

                # Non-streaming stage measurement
                t0 = time.perf_counter()

                docs = pipeline.retriever.invoke(question)

                context = [
                    doc.page_content
                    for doc in docs
                ]

                t1 = time.perf_counter()

                from src.generator import generate

                answer = generate(question, context)

                t2 = time.perf_counter()

                retrieval_ms.append(
                    (t1 - t0) * 1000
                )

                generation_ms.append(
                    (t2 - t1) * 1000
                )

            else:

                answer = run_end_to_end(
                    pipeline,
                    question
                )

            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000

            total_ms.append(elapsed_ms)

            answer_lengths.append(
                len(answer or "")
            )

    return {
        "total": total_ms,
        "retrieval": retrieval_ms,
        "generation": generation_ms,
        "ttft": ttft_ms,
        "answer_len": answer_lengths,
    }

# SUMMARIZE RESULTS

def summarize(samples):

    clean = [
        s for s in samples
        if not math.isnan(s)
    ]

    if not clean:
        return {
            "n": 0,
            "mean": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    return {
        "n": len(clean),
        "mean": sum(clean) / len(clean),
        "p50": percentile(clean, 50),
        "p95": percentile(clean, 95),
        "p99": percentile(clean, 99),
        "min": min(clean),
        "max": max(clean),
    }

def print_row(label, s):

    print(
        f"{label:<12} | "
        f"n={s['n']:<3} "
        f"mean={s['mean']:7.1f} "
        f"p50={s['p50']:7.1f} "
        f"p95={s['p95']:7.1f} "
        f"p99={s['p99']:7.1f} "
        f"min={s['min']:7.1f} "
        f"max={s['max']:7.1f}"
    )

def slo_line(label, p95, budget):

    verdict = (
        "PASS"
        if p95 <= budget
        else "FAIL"
    )

    print(
        f"SLO: {label:<22} "
        f"p95 <= {budget:>5} ms "
        f"-> p95 = {p95:7.0f} ms "
        f"[{verdict}]"
    )

# REPORT

def report(results):

    print("\n" + "=" * 90)
    print("RAG APPLICATION LATENCY REPORT")
    print("=" * 90)

    print(
        f"{'Stage':<12} | "
        f"{'Samples':<5} "
        f"{'Mean':>9} "
        f"{'P50':>9} "
        f"{'P95':>9} "
        f"{'P99':>9} "
        f"{'Min':>9} "
        f"{'Max':>9}"
    )

    print("-" * 90)

    total = summarize(results["total"])

    print_row("end-to-end", total)

    if results["ttft"]:
        print_row(
            "TTFT",
            summarize(results["ttft"])
        )

    if results["retrieval"]:
        print_row(
            "retrieval",
            summarize(results["retrieval"])
        )

        print_row(
            "generation",
            summarize(results["generation"])
        )

    avg_len = (
        sum(results["answer_len"])
        / len(results["answer_len"])
    )

    print("-" * 90)

    print(
        f"Average answer length: {avg_len:.0f} characters"
    )

    print("=" * 90)

    slo_line(
        "Full answer",
        total["p95"],
        SLO_P95_MS
    )

    if results["ttft"]:

        ttft_summary = summarize(
            results["ttft"]
        )

        slo_line(
            "First token",
            ttft_summary["p95"],
            SLO_TTFT_P95_MS
        )

    print("=" * 90)


# ENTRY POINT

def main():

    pipeline = RagPipeline(
        fetch_k=20,
        top_k=5
    )

    results = benchmark(pipeline)

    report(results)

if __name__ == "__main__":
    main()