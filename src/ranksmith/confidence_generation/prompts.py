from __future__ import annotations

import json

from ranksmith.confidence_generation.types import (
    AnswerGenerationSample,
    QueryAnswerabilityGenerationSample,
    QueryContextAnswerabilityGenerationSample,
    RelevanceGenerationSample,
)

ANSWER_SYSTEM_PROMPT = (
    "You answer questions using only the provided context. "
    'Return only JSON with an "answer" string.'
)

RELEVANCE_SYSTEM_PROMPT = (
    "You judge document relevance. "
    'Return only JSON with a "judgment" value of "relevant" or "not_relevant".'
)

QUERY_ANSWERABILITY_SYSTEM_PROMPT = (
    "You are a strict JSON answer API. "
    "Answer questions using your parametric knowledge. "
    "Never reason or explain. Return exactly one JSON object and stop."
)

QUERY_CONTEXT_ANSWERABILITY_SYSTEM_PROMPT = (
    "You are a strict JSON answer API. "
    "Answer questions using only the provided context. "
    "Never reason or explain. Return exactly one JSON object and stop."
)


def build_answer_prompt(
    sample: AnswerGenerationSample,
    *,
    no_answer_value: str,
) -> str:
    no_answer_contract = json.dumps(
        {"answer": no_answer_value},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        f"Question:\n{sample.query}\n\n"
        f"Context:\n{sample.context}\n\n"
        "Return JSON exactly like this shape:\n"
        '{"answer":"..."}\n\n'
        "Use only the context. If the context does not contain the answer, "
        f"return {no_answer_contract}."
    )


def build_relevance_prompt(sample: RelevanceGenerationSample) -> str:
    return (
        f"Query:\n{sample.query}\n\n"
        f"Document:\n{sample.document}\n\n"
        "Return JSON exactly as one of these two shapes:\n"
        '{"judgment":"relevant"}\n'
        '{"judgment":"not_relevant"}\n\n'
        'Use "relevant" if the document contains information useful for '
        "answering the query.\n"
        'Use "not_relevant" otherwise.'
    )


def build_query_answerability_prompt(
    sample: QueryAnswerabilityGenerationSample,
    *,
    no_answer_value: str,
) -> str:
    no_answer_contract = json.dumps(
        {"answer": no_answer_value},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        f"Question:\n{sample.query}\n\n"
        "Return JSON only. Valid examples:\n"
        '{"answer":"short answer"}\n'
        f"{no_answer_contract}\n\n"
        "The answer string must be a short answer only, not an explanation. "
        "Do not output any other text. "
        "Answer from your parametric knowledge. If you are not immediately "
        f"certain, return {no_answer_contract}."
    )


def build_query_context_answerability_prompt(
    sample: QueryContextAnswerabilityGenerationSample,
    *,
    no_answer_value: str,
) -> str:
    no_answer_contract = json.dumps(
        {"answer": no_answer_value},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        f"Question:\n{sample.query}\n\n"
        f"Context:\n{sample.context}\n\n"
        "Return JSON only. Valid examples:\n"
        '{"answer":"short answer"}\n'
        f"{no_answer_contract}\n\n"
        "The answer string must be a short answer only, not an explanation. "
        "Do not output any other text. "
        "Use only the context. If the context does not directly contain the answer, "
        f"return {no_answer_contract}."
    )
