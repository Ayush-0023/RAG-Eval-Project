"""
src/generator.py — the GENERATOR component.

Given a query and context (retrieved chunks), produce an answer grounded in
the context. The prompt is faithfulness-first: answer ONLY from the context,
and abstain when the context doesn't contain the answer.

    from src.generator import generate
    answer = generate("what is drift?", ["chunk text 1", "chunk text 2"])
"""


import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("INCEPTION_API_KEY")
GENERATOR_MODEL = os.getenv("INCEPTION_MODEL", "mercury-3")
BASE_URL = "https://api.inceptionlabs.ai/v1"

if not API_KEY:
    raise ValueError("INCEPTION_API_KEY is missing from .env")

llm = ChatOpenAI(
    model=GENERATOR_MODEL,
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0,
)

# faithfulness-first prompt: ground every claim in the context, abstain if unsure
prompt = ChatPromptTemplate.from_template(
    """
You are a helpful teaching assistant for a course on LLM evaluations. Answer the student's question using ONLY the information in the context provided below.

Rules:
- Use only information present in the context. Do not add outside knowledge.
- If the context does not contain enough information to answer, say exactly:
"I don't have enough information in the course material to answer that."
- Keep the answer clear and concise.

Context:
{context}

Question: {question}

Answer:
"""
)

chain = prompt | llm | StrOutputParser()

def generate(query: str, context: list[str]) -> str:
    """Generate a grounded answer using retrieved context."""
    if not query.strip():
        raise ValueError("Query must not be empty.")

    context_text = "\n\n".join(
        chunk for chunk in context if chunk and chunk.strip()
    )

    if not context_text:
        return (
            "I don't have enough information in the course material "
            "to answer that."
        )

    return chain.invoke({
        "question": query,
        "context": context_text,
    })


if __name__ == "__main__":
    context = [
        "Online evaluation tests a system using live production traffic "
        "after deployment."
    ]

    print(generate("What is online evaluation?", context))