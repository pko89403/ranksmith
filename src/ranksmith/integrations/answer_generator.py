from __future__ import annotations

import json
from dataclasses import dataclass

from ranksmith.errors import RerankParseError, RerankProviderError
from ranksmith.integrations.validation import validate_no_answer_value
from ranksmith.model import ModelMessage, ModelProvider, ModelRequest


@dataclass(frozen=True)
class ProviderAnswerGenerator:
    provider: ModelProvider
    no_answer_value: str = "__NO_ANSWER__"

    def __post_init__(self) -> None:
        validate_no_answer_value(self.no_answer_value)

    def answer_query(self, query: str) -> str:
        return self._complete(
            system=(
                "You are a strict JSON answer API for confidence estimation. "
                "Never reason or explain. Return exactly one JSON object and stop."
            ),
            user=(
                f"Question:\n{query}\n\n"
                "Return JSON only. Valid examples:\n"
                '{"answer":"short answer"}\n'
                f"{_answer_contract(self.no_answer_value)}\n\n"
                "The answer string must be a short answer only, not an "
                "explanation. Do not output any other text. "
                "Answer from your parametric knowledge. If you are not "
                "immediately certain, return "
                f"{_answer_contract(self.no_answer_value)}."
            ),
        )

    def answer_with_context(self, query: str, context: str) -> str:
        return self._complete(
            system=(
                "You are a strict JSON answer API for confidence estimation. "
                "Use only the provided context. Never reason or explain. "
                "Return exactly one JSON object and stop."
            ),
            user=(
                f"Question:\n{query}\n\n"
                f"Context:\n{context}\n\n"
                "Return JSON only. Valid examples:\n"
                '{"answer":"short answer"}\n'
                f"{_answer_contract(self.no_answer_value)}\n\n"
                "The answer string must be a short answer only, not an "
                "explanation. Do not output any other text. "
                "Use only the context. If the context does not directly contain "
                "the answer, "
                f"return {_answer_contract(self.no_answer_value)}."
            ),
        )

    def _complete(self, *, system: str, user: str) -> str:
        try:
            response = self.provider.complete(
                ModelRequest(
                    messages=[
                        ModelMessage(role="system", content=system),
                        ModelMessage(role="user", content=user),
                    ],
                    response_format="json_object",
                    temperature=0,
                )
            )
        except RerankProviderError:
            raise
        except Exception as exc:
            message = str(exc)
            if message:
                raise RerankProviderError(
                    f"Answer generation provider failed: {message}"
                ) from exc
            raise RerankProviderError("Answer generation provider failed.") from exc
        return _parse_answer(response.content)


def _parse_answer(content: str) -> str:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RerankParseError("answer response must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise RerankParseError("answer response must be a JSON object")
    answer = parsed.get("answer")
    if not isinstance(answer, str) or answer.strip() == "":
        raise RerankParseError('answer response must contain a non-empty "answer"')
    return answer


def _answer_contract(no_answer_value: str) -> str:
    return json.dumps(
        {"answer": no_answer_value},
        ensure_ascii=False,
        separators=(",", ":"),
    )
