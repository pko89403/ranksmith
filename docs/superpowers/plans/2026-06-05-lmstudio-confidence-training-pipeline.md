# LM Studio Confidence Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local LM Studio path that can generate answerability confidence datasets, train CBDR-ready scorer artifacts, and run CBDR with LM Studio instead of Azure.

**Architecture:** Add LM Studio as a `ModelProvider` under `ranksmith.integrations`, keep answer generation provider-agnostic, and expose thin scripts over the existing confidence generation/training APIs. CBDR remains unchanged as an algorithm; only its runtime answer generator assembly becomes provider-selectable.

**Tech Stack:** Python 3.10+, OpenAI Python SDK, pytest, argparse, existing `ranksmith.confidence_generation`, existing `ranksmith.confidence_training`, existing `CBDRStrategy`.

---

## File Map

- Create `src/ranksmith/integrations/_lmstudio_provider.py`  
  LM Studio OpenAI-compatible `ModelProvider`. Converts ranksmith `json_object` requests to LM Studio `json_schema`.

- Create `src/ranksmith/integrations/_answer_generator.py`  
  Provider-agnostic sync answer generator with `answer_query()` and `answer_with_context()`.

- Modify `src/ranksmith/integrations/_azure_answer_generator.py`  
  Make `AzureAnswerGenerator` reuse `ProviderAnswerGenerator` instead of owning duplicate prompt/parse logic.

- Modify `src/ranksmith/integrations/__init__.py`  
  Export `LMStudioModelProvider` and `ProviderAnswerGenerator` from submodule only.

- Create `scripts/generate_confidence_dataset.py`  
  CLI for `query_answerability_confidence` and `query_context_answerability_confidence`.

- Create `scripts/train_confidence_scorer.py`  
  CLI wrapper for `train_confidence_scorer()`.

- Create `src/ranksmith/confidence_training/_dataset_report.py`  
  Dataset balance/source/group summary helper.

- Create `scripts/report_confidence_dataset.py`  
  CLI wrapper for dataset report.

- Modify `scripts/compare_reranking.py`  
  Add `--cbdr-answer-provider azure|lmstudio` and LM Studio flags for CBDR runtime.

- Create fixtures:
  - `tests/fixtures/confidence_query_answerability_raw.jsonl`
  - `tests/fixtures/confidence_query_context_answerability_raw.jsonl`

- Add tests:
  - `tests/test_lmstudio_provider.py`
  - `tests/test_provider_answer_generator.py`
  - `tests/test_generate_confidence_dataset_script.py`
  - `tests/test_train_confidence_scorer_script.py`
  - `tests/test_confidence_training_dataset_report.py`
  - extend `tests/test_compare_reranking.py`

---

## Task 1: LM Studio Provider

**Files:**
- Create: `src/ranksmith/integrations/_lmstudio_provider.py`
- Modify: `src/ranksmith/integrations/__init__.py`
- Test: `tests/test_lmstudio_provider.py`

- [x] **Step 1: Write failing provider tests**

Create `tests/test_lmstudio_provider.py`:

```python
from __future__ import annotations

import importlib

import pytest

from ranksmith.errors import RerankInputError, RerankProviderError
from ranksmith.model import ModelMessage, ModelRequest


class FakeChoiceMessage:
    content = '{"answer":"Paris"}'


class FakeChoice:
    message = FakeChoiceMessage()


class FakeUsage:
    prompt_tokens = 3
    completion_tokens = 2
    total_tokens = 5


class FakeResponse:
    choices = [FakeChoice()]
    usage = FakeUsage()


class FakeCompletions:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs: object) -> FakeResponse:
        self.kwargs = kwargs
        return FakeResponse()


class FakeChat:
    def __init__(self) -> None:
        self.completions = FakeCompletions()


class FakeClient:
    def __init__(self) -> None:
        self.chat = FakeChat()


def test_lmstudio_provider_is_submodule_export_only() -> None:
    integrations = importlib.import_module("ranksmith.integrations")
    root = importlib.import_module("ranksmith")

    assert integrations.LMStudioModelProvider is not None
    assert not hasattr(root, "LMStudioModelProvider")


def test_lmstudio_provider_converts_json_object_to_json_schema() -> None:
    from ranksmith.integrations import LMStudioModelProvider

    client = FakeClient()
    provider = LMStudioModelProvider(model="google/gemma-4-12b", client=client)

    response = provider.complete(
        ModelRequest(
            messages=[ModelMessage(role="user", content="Return JSON.")],
            response_format="json_object",
            temperature=0,
        )
    )

    kwargs = client.chat.completions.kwargs
    assert response.content == '{"answer":"Paris"}'
    assert kwargs["model"] == "google/gemma-4-12b"
    assert kwargs["temperature"] == 0
    assert kwargs["max_tokens"] == 128
    assert kwargs["response_format"]["type"] == "json_schema"
    assert kwargs["response_format"]["json_schema"]["name"] == "ranksmith_json_response"


def test_lmstudio_provider_uses_env_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from ranksmith.integrations import LMStudioModelProvider

    monkeypatch.setenv("LMSTUDIO_MODEL", "google/gemma-4-12b")

    provider = LMStudioModelProvider(client=FakeClient())

    provider.complete(ModelRequest(messages=[ModelMessage(role="user", content="x")]))


def test_lmstudio_provider_requires_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from ranksmith.integrations import LMStudioModelProvider

    monkeypatch.delenv("LMSTUDIO_MODEL", raising=False)

    with pytest.raises(RerankInputError, match="LMSTUDIO_MODEL"):
        LMStudioModelProvider(client=FakeClient())


def test_lmstudio_provider_wraps_client_error() -> None:
    from ranksmith.integrations import LMStudioModelProvider

    class FailingCompletions:
        def create(self, **kwargs: object) -> object:
            del kwargs
            raise RuntimeError("server down")

    class FailingChat:
        completions = FailingCompletions()

    class FailingClient:
        chat = FailingChat()

    provider = LMStudioModelProvider(model="google/gemma-4-12b", client=FailingClient())

    with pytest.raises(RerankProviderError, match="server down"):
        provider.complete(ModelRequest(messages=[ModelMessage(role="user", content="x")]))
```

- [x] **Step 2: Run failing test**

```bash
uv run pytest tests/test_lmstudio_provider.py -q
```

Expected: fails because `LMStudioModelProvider` does not exist.

- [x] **Step 3: Implement provider**

Create `src/ranksmith/integrations/_lmstudio_provider.py`:

```python
from __future__ import annotations

import os
from typing import Any, cast

from openai import OpenAI

from ranksmith.errors import RerankInputError, RerankProviderError
from ranksmith.model import ModelRequest, ModelResponse
from ranksmith.types import RerankUsage


class LMStudioModelProvider:
    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        max_tokens: int = 128,
        client: Any | None = None,
    ) -> None:
        resolved_model = model or _env_value("LMSTUDIO_MODEL")
        if resolved_model is None or resolved_model == "":
            raise RerankInputError("LMSTUDIO_MODEL is required")
        if max_tokens < 1:
            raise RerankInputError("max_tokens must be greater than 0")
        self._model = resolved_model
        self._max_tokens = max_tokens
        self._client = client or OpenAI(
            base_url=base_url or _env_value("LMSTUDIO_BASE_URL") or "http://localhost:1234/v1",
            api_key=api_key or _env_value("LMSTUDIO_API_KEY") or "lm-studio",
            timeout=timeout,
        )

    def complete(self, request: ModelRequest) -> ModelResponse:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=_to_openai_messages(request),
                response_format=_lmstudio_response_format(request),
                temperature=request.temperature,
                max_tokens=self._max_tokens,
            )
        except Exception as exc:
            raise RerankProviderError(str(exc)) from exc

        content = _extract_content(response)
        if content is None or content == "":
            raise RerankProviderError("LM Studio returned an empty response.")
        return ModelResponse(content=content, usage=_extract_usage(response))


def _lmstudio_response_format(request: ModelRequest) -> dict[str, object]:
    if request.response_format != "json_object":
        raise RerankProviderError("LM Studio provider only supports json_object requests.")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "ranksmith_json_response",
            "schema": {"type": "object"},
        },
    }


def _to_openai_messages(request: ModelRequest) -> list[dict[str, str]]:
    return [
        {"role": message.role, "content": message.content}
        for message in request.messages
    ]


def _extract_content(response: object) -> str | None:
    choices = getattr(response, "choices", None)
    if not choices:
        raise RerankProviderError("LM Studio returned an invalid response.")
    try:
        return cast(str | None, choices[0].message.content)
    except AttributeError as exc:
        raise RerankProviderError("LM Studio returned an invalid response.") from exc


def _extract_usage(response: object) -> RerankUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return RerankUsage(
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
    )


def _env_value(name: str) -> str | None:
    value = os.environ.get(name)
    if value is not None and value != "":
        return value
    return None
```

Modify `src/ranksmith/integrations/__init__.py`:

```python
from ranksmith.integrations._azure_answer_generator import AzureAnswerGenerator
from ranksmith.integrations._lmstudio_provider import LMStudioModelProvider

__all__ = ["AzureAnswerGenerator", "LMStudioModelProvider"]
```

- [x] **Step 4: Run provider tests**

```bash
uv run pytest tests/test_lmstudio_provider.py -q
```

Expected: all tests pass.

---

## Task 2: Provider-Agnostic Answer Generator

**Files:**
- Create: `src/ranksmith/integrations/_answer_generator.py`
- Modify: `src/ranksmith/integrations/_azure_answer_generator.py`
- Modify: `src/ranksmith/integrations/__init__.py`
- Test: `tests/test_provider_answer_generator.py`
- Update: `tests/test_azure_answer_generator.py`

- [x] **Step 1: Write failing tests**

Create `tests/test_provider_answer_generator.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import pytest

from ranksmith.errors import RerankParseError, RerankProviderError
from ranksmith.model import ModelRequest, ModelResponse


@dataclass
class FakeProvider:
    responses: list[str]
    requests: list[ModelRequest]

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(content=self.responses.pop(0))


def test_provider_answer_generator_parses_query_answer() -> None:
    from ranksmith.integrations import ProviderAnswerGenerator

    provider = FakeProvider(responses=['{"answer":"Paris"}'], requests=[])
    generator = ProviderAnswerGenerator(provider=provider)

    assert generator.answer_query("capital of france?") == "Paris"
    assert provider.requests[0].response_format == "json_object"
    assert provider.requests[0].temperature == 0


def test_provider_answer_generator_parses_context_answer() -> None:
    from ranksmith.integrations import ProviderAnswerGenerator

    provider = FakeProvider(responses=['{"answer":"Nancy Travis"}'], requests=[])
    generator = ProviderAnswerGenerator(provider=provider)

    assert generator.answer_with_context("who?", "Nancy Travis played Karen.") == "Nancy Travis"
    assert "Use only the context" in provider.requests[0].messages[1].content


@pytest.mark.parametrize("content", ["not json", "{}", '{"answer":""}', '{"answer":1}'])
def test_provider_answer_generator_rejects_invalid_json(content: str) -> None:
    from ranksmith.integrations import ProviderAnswerGenerator

    generator = ProviderAnswerGenerator(
        provider=FakeProvider(responses=[content], requests=[]),
    )

    with pytest.raises(RerankParseError):
        generator.answer_query("x")


def test_provider_answer_generator_preserves_provider_error() -> None:
    from ranksmith.integrations import ProviderAnswerGenerator

    class FailingProvider:
        def complete(self, request: ModelRequest) -> ModelResponse:
            del request
            raise RerankProviderError("provider failed")

    generator = ProviderAnswerGenerator(provider=FailingProvider())

    with pytest.raises(RerankProviderError, match="provider failed"):
        generator.answer_query("x")
```

- [x] **Step 2: Run failing test**

```bash
uv run pytest tests/test_provider_answer_generator.py -q
```

Expected: fails because `ProviderAnswerGenerator` does not exist.

- [x] **Step 3: Implement generator**

Create `src/ranksmith/integrations/_answer_generator.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass

from ranksmith.errors import RerankParseError, RerankProviderError
from ranksmith.model import ModelMessage, ModelProvider, ModelRequest


@dataclass(frozen=True)
class ProviderAnswerGenerator:
    provider: ModelProvider
    no_answer_value: str = "__NO_ANSWER__"

    def __post_init__(self) -> None:
        _validate_no_answer_value(self.no_answer_value)

    def answer_query(self, query: str) -> str:
        return self._complete(
            system=(
                "You answer questions for confidence estimation. Return only JSON "
                'with an "answer" string.'
            ),
            user=(
                f"Question:\n{query}\n\n"
                "Return JSON exactly like this shape:\n"
                '{"answer":"..."}\n\n'
                "Answer from your parametric knowledge. If you do not know the "
                f"answer, return {_answer_contract(self.no_answer_value)}."
            ),
        )

    def answer_with_context(self, query: str, context: str) -> str:
        return self._complete(
            system=(
                "You answer questions using the provided context for confidence "
                'estimation. Return only JSON with an "answer" string.'
            ),
            user=(
                f"Question:\n{query}\n\n"
                f"Context:\n{context}\n\n"
                "Return JSON exactly like this shape:\n"
                '{"answer":"..."}\n\n'
                "Use only the context. If the context does not contain the answer, "
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
            raise RerankProviderError(str(exc)) from exc
        return parse_answer(response.content)


def parse_answer(content: str) -> str:
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


def _validate_no_answer_value(value: object) -> None:
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError("no_answer_value must be a non-empty string")
```

Modify `src/ranksmith/integrations/_azure_answer_generator.py` so `AzureAnswerGenerator` delegates:

```python
from ranksmith.integrations._answer_generator import ProviderAnswerGenerator

# keep env helpers and Azure provider assembly
# replace answer_query/answer_with_context/_complete/_parse_answer with:

    def answer_query(self, query: str) -> str:
        return ProviderAnswerGenerator(
            provider=self.provider,
            no_answer_value=self.no_answer_value,
        ).answer_query(query)

    def answer_with_context(self, query: str, context: str) -> str:
        return ProviderAnswerGenerator(
            provider=self.provider,
            no_answer_value=self.no_answer_value,
        ).answer_with_context(query, context)
```

Modify `src/ranksmith/integrations/__init__.py`:

```python
from ranksmith.integrations._answer_generator import ProviderAnswerGenerator
from ranksmith.integrations._azure_answer_generator import AzureAnswerGenerator
from ranksmith.integrations._lmstudio_provider import LMStudioModelProvider

__all__ = [
    "AzureAnswerGenerator",
    "LMStudioModelProvider",
    "ProviderAnswerGenerator",
]
```

- [x] **Step 4: Run tests**

```bash
uv run pytest tests/test_provider_answer_generator.py tests/test_azure_answer_generator.py -q
```

Expected: all tests pass.

---

## Task 3: Generation CLI

**Files:**
- Create: `scripts/generate_confidence_dataset.py`
- Create: `tests/fixtures/confidence_query_answerability_raw.jsonl`
- Create: `tests/fixtures/confidence_query_context_answerability_raw.jsonl`
- Test: `tests/test_generate_confidence_dataset_script.py`

- [x] **Step 1: Add raw fixtures**

Create `tests/fixtures/confidence_query_answerability_raw.jsonl`:

```jsonl
{"id":"qa-1","query":"What is the capital of France?","gold_answer":"Paris","source":"fixture-general","group_id":"q1"}
{"id":"qa-2","query":"Who played Karen in Married to the Mob?","gold_answer":"Nancy Travis","source":"fixture-movie","group_id":"q2"}
```

Create `tests/fixtures/confidence_query_context_answerability_raw.jsonl`:

```jsonl
{"id":"qc-1","query":"What is the capital of France?","context":"Paris is the capital and largest city of France.","gold_answer":"Paris","source":"fixture-general","group_id":"q1"}
{"id":"qc-2","query":"Who played Karen in Married to the Mob?","context":"Nancy Travis played Karen Lutnick in Married to the Mob.","gold_answer":"Nancy Travis","source":"fixture-movie","group_id":"q2"}
```

- [x] **Step 2: Write failing CLI tests**

Create `tests/test_generate_confidence_dataset_script.py`:

```python
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path("scripts/generate_confidence_dataset.py")


def _load_script() -> object:
    spec = importlib.util.spec_from_file_location("generate_confidence_dataset", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_args_accepts_query_answerability() -> None:
    module = _load_script()

    args = module.parse_args(
        [
            "--task",
            "query_answerability_confidence",
            "--provider",
            "lmstudio",
            "--lmstudio-model",
            "google/gemma-4-12b",
            "--input",
            "in.jsonl",
            "--output",
            "out.jsonl",
        ]
    )

    assert args.task == "query_answerability_confidence"
    assert args.lmstudio_model == "google/gemma-4-12b"


def test_generation_cli_runs_with_fake_provider(tmp_path: Path, monkeypatch) -> None:
    module = _load_script()
    input_path = tmp_path / "raw.jsonl"
    output_path = tmp_path / "canonical.jsonl"
    input_path.write_text(
        '{"id":"1","query":"What is the capital of France?","gold_answer":"Paris"}\n',
        encoding="utf-8",
    )

    class FakeProvider:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def complete(self, request):
            del request
            from ranksmith.model import ModelResponse

            return ModelResponse(content='{"answer":"Paris"}')

    monkeypatch.setattr(module, "LMStudioModelProvider", FakeProvider)

    result = module.run(
        module.parse_args(
            [
                "--task",
                "query_answerability_confidence",
                "--provider",
                "lmstudio",
                "--lmstudio-model",
                "google/gemma-4-12b",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--overwrite",
            ]
        )
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert result.generated_count == 1
    assert rows[0]["task_type"] == "query_answerability_confidence"
    assert rows[0]["label"] == 1
```

- [x] **Step 3: Implement CLI**

Create `scripts/generate_confidence_dataset.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ranksmith.confidence_generation import (
    QueryAnswerabilityGenerationConfig,
    QueryContextAnswerabilityGenerationConfig,
    generate_query_answerability_confidence_dataset,
    generate_query_context_answerability_confidence_dataset,
)
from ranksmith.integrations import LMStudioModelProvider


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result.__dict__ | {"output_path": str(result.output_path)}, sort_keys=True))


def run(args: argparse.Namespace):
    provider = LMStudioModelProvider(
        base_url=args.lmstudio_base_url,
        model=args.lmstudio_model,
        api_key=args.lmstudio_api_key,
        timeout=args.timeout,
        max_tokens=args.lmstudio_max_tokens,
    )
    if args.task == "query_answerability_confidence":
        return generate_query_answerability_confidence_dataset(
            QueryAnswerabilityGenerationConfig(
                input_path=args.input,
                output_path=args.output,
                provider=provider,
                overwrite=args.overwrite,
                resume=args.resume,
                max_items=args.max_items,
                source=args.source,
            )
        )
    if args.task == "query_context_answerability_confidence":
        return generate_query_context_answerability_confidence_dataset(
            QueryContextAnswerabilityGenerationConfig(
                input_path=args.input,
                output_path=args.output,
                provider=provider,
                overwrite=args.overwrite,
                resume=args.resume,
                max_items=args.max_items,
                max_context_chars=args.max_context_chars,
                source=args.source,
            )
        )
    raise SystemExit(f"unsupported task: {args.task}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("query_answerability_confidence", "query_context_answerability_confidence"), required=True)
    parser.add_argument("--provider", choices=("lmstudio",), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--source")
    parser.add_argument("--max-context-chars", type=int, default=8000)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--lmstudio-base-url")
    parser.add_argument("--lmstudio-model")
    parser.add_argument("--lmstudio-api-key")
    parser.add_argument("--lmstudio-max-tokens", type=int, default=128)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
```

- [x] **Step 4: Run CLI tests**

```bash
uv run pytest tests/test_generate_confidence_dataset_script.py -q
```

Expected: all tests pass.

---

## Task 4: Training CLI

**Files:**
- Create: `scripts/train_confidence_scorer.py`
- Test: `tests/test_train_confidence_scorer_script.py`

- [x] **Step 1: Write failing CLI test**

Create `tests/test_train_confidence_scorer_script.py`:

```python
from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path("scripts/train_confidence_scorer.py")


def _load_script() -> object:
    spec = importlib.util.spec_from_file_location("train_confidence_scorer_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_cli_builds_config(monkeypatch, tmp_path: Path) -> None:
    module = _load_script()
    captured = {}

    def fake_train(config):
        captured["config"] = config
        from ranksmith.confidence_training import ConfidenceTrainingResult

        return ConfidenceTrainingResult(
            output_dir=Path(config.output_dir),
            export_path=Path(config.export_path),
            report_path=Path(config.output_dir) / "report.json",
            metadata_path=Path(config.output_dir) / "metadata.json",
        )

    monkeypatch.setattr(module, "train_confidence_scorer", fake_train)

    module.run(
        module.parse_args(
            [
                "--task",
                "query_answerability_confidence",
                "--dataset",
                str(tmp_path / "data.jsonl"),
                "--output-dir",
                str(tmp_path / "out"),
                "--export-path",
                str(tmp_path / "artifact.joblib"),
                "--encoder-name",
                "bert-base-uncased",
                "--max-length",
                "256",
            ]
        )
    )

    config = captured["config"]
    assert config.task_type == "query_answerability_confidence"
    assert config.encoder_name == "bert-base-uncased"
    assert config.max_length == 256
```

- [x] **Step 2: Implement CLI**

Create `scripts/train_confidence_scorer.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ranksmith.confidence_training import (
    ConfidenceTrainingConfig,
    train_confidence_scorer,
)


def main() -> None:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "export_path": str(result.export_path),
                "report_path": str(result.report_path),
                "metadata_path": str(result.metadata_path),
            },
            sort_keys=True,
        )
    )


def run(args: argparse.Namespace):
    return train_confidence_scorer(
        ConfidenceTrainingConfig(
            task_type=args.task,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            export_path=args.export_path,
            encoder_name=args.encoder_name,
            encoder_revision=args.encoder_revision,
            tokenizer_name=args.tokenizer_name,
            tokenizer_revision=args.tokenizer_revision,
            cache_dir=str(args.cache_dir) if args.cache_dir is not None else None,
            local_files_only=args.local_files_only,
            max_length=args.max_length,
            allow_truncation=args.allow_truncation,
            seed=args.seed,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            test_ratio=args.test_ratio,
        )
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("query_answerability_confidence", "query_context_answerability_confidence"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--export-path", type=Path, required=True)
    parser.add_argument("--encoder-name", default="bert-base-uncased")
    parser.add_argument("--encoder-revision")
    parser.add_argument("--tokenizer-name")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--allow-truncation", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
```

- [x] **Step 3: Run test**

```bash
uv run pytest tests/test_train_confidence_scorer_script.py -q
```

Expected: all tests pass.

---

## Task 5: Dataset Balance Report

**Files:**
- Create: `src/ranksmith/confidence_training/_dataset_report.py`
- Create: `scripts/report_confidence_dataset.py`
- Test: `tests/test_confidence_training_dataset_report.py`

- [x] **Step 1: Write failing report tests**

Create `tests/test_confidence_training_dataset_report.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from ranksmith.confidence_training._dataset_report import build_dataset_report


def test_build_dataset_report_counts_sources_and_groups(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"id":"1","task_type":"query_answerability_confidence","query":"q","answer":"a","label":1,"source":"s1","group_id":"g1"}),
                json.dumps({"id":"2","task_type":"query_answerability_confidence","query":"q","answer":"x","label":0,"source":"s1","group_id":"g1"}),
                json.dumps({"id":"3","task_type":"query_answerability_confidence","query":"q","answer":"a","label":1,"source":"s2"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_dataset_report(path, task_type="query_answerability_confidence")

    assert report["sample_count"] == 3
    assert report["positive_count"] == 2
    assert report["negative_count"] == 1
    assert report["missing_group_id_count"] == 1
    assert report["sources"]["s1"]["sample_count"] == 2
    assert report["sources"]["s1"]["positive_count"] == 1
```

- [x] **Step 2: Implement report helper**

Create `src/ranksmith/confidence_training/_dataset_report.py`:

```python
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ranksmith.confidence import TaskType
from ranksmith.confidence_training._dataset import load_canonical_dataset


def build_dataset_report(path: str | Path, *, task_type: TaskType) -> dict[str, Any]:
    samples = load_canonical_dataset(path, task_type=task_type)
    positive_count = sum(sample.label for sample in samples)
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    missing_source_count = 0
    missing_group_id_count = 0
    group_ids: set[str] = set()

    for sample in samples:
        source = sample.source
        if source is None:
            missing_source_count += 1
            source = "__MISSING__"
        if sample.group_id is None:
            missing_group_id_count += 1
        else:
            group_ids.add(sample.group_id)
        source_counts[source]["sample_count"] += 1
        source_counts[source]["positive_count"] += sample.label
        source_counts[source]["negative_count"] += 1 - sample.label

    return {
        "task_type": task_type,
        "sample_count": len(samples),
        "positive_count": positive_count,
        "negative_count": len(samples) - positive_count,
        "positive_rate": _rate(positive_count, len(samples)),
        "source_count": len(source_counts),
        "group_count": len(group_ids),
        "missing_source_count": missing_source_count,
        "missing_group_id_count": missing_group_id_count,
        "sources": {
            source: {
                "sample_count": counts["sample_count"],
                "positive_count": counts["positive_count"],
                "negative_count": counts["negative_count"],
                "positive_rate": _rate(counts["positive_count"], counts["sample_count"]),
            }
            for source, counts in sorted(source_counts.items())
        },
    }


def _rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return count / total
```

- [x] **Step 3: Add CLI**

Create `scripts/report_confidence_dataset.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ranksmith.confidence_training._dataset_report import build_dataset_report


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def run(args: argparse.Namespace) -> dict[str, object]:
    return build_dataset_report(args.dataset, task_type=args.task)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("query_answerability_confidence", "query_context_answerability_confidence"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
```

- [x] **Step 4: Run report tests**

```bash
uv run pytest tests/test_confidence_training_dataset_report.py -q
```

Expected: all tests pass.

---

## Task 6: CBDR Benchmark LM Studio Runtime

**Files:**
- Modify: `scripts/compare_reranking.py`
- Test: extend `tests/test_compare_reranking.py`

- [x] **Step 1: Add failing compare test**

Add to `tests/test_compare_reranking.py`:

```python
def test_compare_cbdr_can_use_lmstudio_answer_provider(monkeypatch, tmp_path):
    import scripts.compare_reranking as compare_reranking
    from ranksmith._benchmark import BenchmarkCase, BenchmarkDocument

    created = {}

    class FakeLMStudioProvider:
        def __init__(self, **kwargs):
            created["lmstudio"] = kwargs

    class FakeProviderAnswerGenerator:
        def __init__(self, *, provider, no_answer_value="__NO_ANSWER__"):
            created["generator"] = provider
            self.no_answer_value = no_answer_value

    class FakeCBDRStrategy:
        @classmethod
        def from_artifacts(cls, **kwargs):
            created["strategy"] = kwargs
            return cls()

        def rerank(self, *, query, documents, top_k=None):
            del query, top_k
            return [
                type("Result", (), {"document": document})
                for document in documents
            ]

    monkeypatch.setattr("ranksmith.integrations.LMStudioModelProvider", FakeLMStudioProvider)
    monkeypatch.setattr("ranksmith.integrations.ProviderAnswerGenerator", FakeProviderAnswerGenerator)
    monkeypatch.setattr("ranksmith.strategies.CBDRStrategy", FakeCBDRStrategy)

    case = BenchmarkCase(
        fixture_id="fixture",
        dataset="dataset",
        source="source",
        license="license",
        query_id="q1",
        query="query",
        documents=(BenchmarkDocument(id="d1", text="doc", title=""),),
        qrels={"d1": 1},
    )

    ranking = compare_reranking._rank_case(
        case=case,
        algorithm="cbdr",
        window_size=20,
        stride=10,
        passes=10,
        tourrank_rounds=2,
        set_size=3,
        cbdr_base_artifact=tmp_path / "base.joblib",
        cbdr_context_artifact=tmp_path / "context.joblib",
        cbdr_answer_provider="lmstudio",
        lmstudio_model="google/gemma-4-12b",
    )

    assert ranking == ("d1",)
    assert created["lmstudio"]["model"] == "google/gemma-4-12b"
    assert created["strategy"]["answer_generator"].no_answer_value == "__NO_ANSWER__"
```

- [x] **Step 2: Modify compare runner**

In `scripts/compare_reranking.py`:

Add parser flags:

```python
parser.add_argument("--cbdr-answer-provider", choices=("azure", "lmstudio"), default="azure")
parser.add_argument("--lmstudio-base-url")
parser.add_argument("--lmstudio-model")
parser.add_argument("--lmstudio-api-key")
parser.add_argument("--lmstudio-max-tokens", type=int, default=128)
```

Add `_rank_case(...)` parameters:

```python
cbdr_answer_provider: str = "azure",
lmstudio_base_url: str | None = None,
lmstudio_model: str | None = None,
lmstudio_api_key: str | None = None,
lmstudio_max_tokens: int = 128,
```

Replace hardcoded CBDR generator:

```python
if cbdr_answer_provider == "azure":
    answer_generator = AzureAnswerGenerator.from_env(timeout=timeout)
elif cbdr_answer_provider == "lmstudio":
    from ranksmith.integrations import LMStudioModelProvider, ProviderAnswerGenerator

    answer_generator = ProviderAnswerGenerator(
        provider=LMStudioModelProvider(
            base_url=lmstudio_base_url,
            model=lmstudio_model,
            api_key=lmstudio_api_key,
            timeout=timeout,
            max_tokens=lmstudio_max_tokens,
        )
    )
else:
    raise SystemExit("--cbdr-answer-provider must be azure or lmstudio.")
```

Pass new args from the caller to `_rank_case(...)`.

- [x] **Step 3: Run compare tests**

```bash
uv run pytest tests/test_compare_reranking.py -q
```

Expected: all tests pass.

---

## Task 7: Docs And Spec Checklist

**Files:**
- Modify: `docs/specs/spec_lmstudio_confidence_training_pipeline.md`
- Modify: `README.md`
- Modify: `README.ko.md`
- Modify: `docs/wiki/02_architecture.md`

- [x] **Step 1: Update spec checklist**

Mark implemented items as `[x]` only after the corresponding tests pass.

- [x] **Step 2: Update architecture wiki**

Add under `Integrations`:

```markdown
- `LMStudioModelProvider`: LM Studio OpenAI-compatible local model provider for confidence dataset generation and CBDR answer generation.
- `ProviderAnswerGenerator`: provider-agnostic sync answer helper used by Azure and LM Studio runtime paths.
```

Add under `Confidence`:

```markdown
Confidence generation/training can be operated through local CLI scripts for query-only and query+context answerability scorer artifacts. The scripts are utility entry points and do not add a new reranking algorithm.
```

- [x] **Step 3: Update README and README.ko**

Add a minimal LM Studio confidence pipeline section with:

```bash
lms server start

uv run python scripts/generate_confidence_dataset.py \
  --task query_answerability_confidence \
  --provider lmstudio \
  --lmstudio-model google/gemma-4-12b \
  --input runs/confidence/local/raw/query_answerability.jsonl \
  --output runs/confidence/local/canonical/query_answerability_confidence.jsonl \
  --resume
```

Do not add benchmark quality numbers.

---

## Task 8: Verification

**Files:** all touched files.

- [x] **Step 1: Run targeted tests**

```bash
uv run pytest \
  tests/test_lmstudio_provider.py \
  tests/test_provider_answer_generator.py \
  tests/test_azure_answer_generator.py \
  tests/test_generate_confidence_dataset_script.py \
  tests/test_train_confidence_scorer_script.py \
  tests/test_confidence_training_dataset_report.py \
  tests/test_compare_reranking.py \
  -q
```

Expected: all selected tests pass.

- [x] **Step 2: Run full verification**

```bash
./scripts/verify.sh
```

Expected: ruff, format check, mypy, pytest, and build pass.

- [x] **Step 3: Optional live LM Studio smoke**

Requires LM Studio server running and `google/gemma-4-12b` loaded.

```bash
lms server status

uv run python scripts/generate_confidence_dataset.py \
  --task query_answerability_confidence \
  --provider lmstudio \
  --lmstudio-model google/gemma-4-12b \
  --input tests/fixtures/confidence_query_answerability_raw.jsonl \
  --output /tmp/ranksmith-lmstudio-query-answerability.jsonl \
  --overwrite \
  --max-items 2
```

Expected: canonical JSONL is written with strict `answer` values and labels.

---

## Self-Review

- Spec coverage:
  - LM Studio provider: Task 1.
  - `json_object -> json_schema`: Task 1.
  - provider-agnostic answer generation and CBDR LM Studio runtime: Tasks 2 and 6.
  - generation CLI: Task 3.
  - training CLI: Task 4.
  - dataset balance report: Task 5.
  - docs/spec/wiki/README: Task 7.
  - verification/live smoke: Task 8.

- Known intentional limits:
  - No automatic external dataset download.
  - No async generation.
  - No model fine-tuning.
  - No benchmark quality numbers without real summary artifacts.
