from __future__ import annotations

from ranksmith.confidence_generation.types import (
    RelevanceGenerationSample,
)

RELEVANCE_SYSTEM_PROMPT = (
    "You judge document relevance. "
    'Return only JSON with a "judgment" value of "relevant" or "not_relevant".'
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
