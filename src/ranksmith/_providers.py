from __future__ import annotations

from typing import Protocol

from openai import AzureOpenAI

from ranksmith.errors import RerankProviderError
from ranksmith.types import Document


class LLMProvider(Protocol):
    def rank(self, query: str, documents: list[Document]) -> str:
        """Return a JSON string containing a 1-based ranking permutation."""


class AzureOpenAIProvider:
    def __init__(
        self,
        *,
        api_key: str,
        azure_endpoint: str,
        azure_deployment: str,
        api_version: str,
        timeout: float | None = None,
    ) -> None:
        self._azure_deployment = azure_deployment
        self._client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            timeout=timeout,
        )

    def rank(self, query: str, documents: list[Document]) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._azure_deployment,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a reranking engine. Return only JSON with "
                            'a "ranking" array. The ranking must be a permutation '
                            "of the candidate numbers."
                        ),
                    },
                    {"role": "user", "content": _build_prompt(query, documents)},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
        except Exception as exc:
            raise RerankProviderError(str(exc)) from exc

        content = response.choices[0].message.content
        if content is None:
            raise RerankProviderError("Azure OpenAI returned an empty response.")
        return content


def _build_prompt(query: str, documents: list[Document]) -> str:
    candidates = "\n\n".join(
        [
            f"[{index}]\n{document.text}"
            for index, document in enumerate(documents, start=1)
        ]
    )
    return (
        "Rank the candidate documents by relevance to the query.\n\n"
        f"Query:\n{query}\n\n"
        f"Candidate documents:\n{candidates}\n\n"
        "Return JSON exactly like this shape:\n"
        '{"ranking": [1, 2, 3]}\n'
        "Use each candidate number exactly once."
    )
