# Pairwise Ranking Prompting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `PairwiseStrategy` / `AsyncPairwiseStrategy` with `prp_sliding_k` ranking, strict pairwise JSON parsing, Azure provider support, and benchmark comparison support.

**Architecture:** Keep listwise and pairwise provider contracts separate. `AzureOpenAIProvider` implements both `rank()` and `compare()` so the existing reranker injection path keeps working. Pairwise strategy owns PRP-Sliding-K traversal and returns the same `RerankResult` contract as listwise.

**Tech Stack:** Python 3.10+, dataclasses, Protocols, pytest, Azure OpenAI SDK, existing `uv` verification scripts.

---

### Task 1: Sync Pairwise Strategy

**Files:**
- Modify: `tests/test_ranksmith.py`
- Modify: `src/ranksmith/_providers.py`
- Modify: `src/ranksmith/strategies.py`
- Modify: `src/ranksmith/__init__.py`

- [ ] **Step 1: Write failing sync tests**

Add tests for:
- `PairwiseStrategy()` default `passes == 10`
- `prp_sliding_k` calls A/B and B/A from right to left
- consistent preference swaps documents
- conflicting preference keeps current order
- invalid pairwise JSON raises `RerankParseError`
- provider without `compare()` raises `RerankInputError`

- [ ] **Step 2: Verify tests fail**

Run:

```bash
uv run pytest tests/test_ranksmith.py -q
```

Expected: tests fail because `PairwiseStrategy` is not importable or not implemented.

- [ ] **Step 3: Implement sync provider protocols and strategy**

Add:
- `PairwiseLLMProvider.compare(query, document_a, document_b) -> str`
- `PairwiseStrategy(algorithm="prp_sliding_k", passes=10, max_document_chars=4000)`
- `_parse_pairwise_winner(raw_response)`
- provider contract fast fail with `RerankInputError`
- export `PairwiseStrategy` from `ranksmith`

- [ ] **Step 4: Verify sync tests pass**

Run:

```bash
uv run pytest tests/test_ranksmith.py -q
```

Expected: all sync tests pass.

### Task 2: Async Pairwise Strategy

**Files:**
- Modify: `tests/test_async_providers.py`
- Modify: `src/ranksmith/_providers.py`
- Modify: `src/ranksmith/strategies.py`
- Modify: `src/ranksmith/__init__.py`

- [ ] **Step 1: Write failing async tests**

Add tests for:
- async PRP-Sliding-K order
- async A/B and B/A call order
- async invalid pairwise JSON fast fail
- async provider without `compare()` fast fail

- [ ] **Step 2: Verify tests fail**

Run:

```bash
uv run pytest tests/test_async_providers.py -q
```

Expected: tests fail because async pairwise behavior is not implemented.

- [ ] **Step 3: Implement async protocol and strategy**

Add:
- `AsyncPairwiseLLMProvider.compare(...)`
- `AsyncPairwiseStrategy`
- async pairwise provider validation
- export `AsyncPairwiseStrategy`

- [ ] **Step 4: Verify async tests pass**

Run:

```bash
uv run pytest tests/test_async_providers.py -q
```

Expected: all async tests pass.

### Task 3: Azure Pairwise Provider

**Files:**
- Modify: `src/ranksmith/_providers.py`
- Modify: `tests/test_usage_hook.py`

- [ ] **Step 1: Write failing provider tests**

Add tests that verify:
- pairwise prompt asks for `{"winner": "A"}`
- `compare()` emits usage through the existing usage callback
- async `compare()` emits usage through async callback

- [ ] **Step 2: Verify tests fail**

Run:

```bash
uv run pytest tests/test_usage_hook.py -q
```

Expected: tests fail because `compare()` does not exist.

- [ ] **Step 3: Implement Azure compare methods**

Add:
- `_build_pairwise_prompt(query, document_a, document_b)`
- `AzureOpenAIProvider.compare(...)`
- `AsyncAzureOpenAIProvider.compare(...)`

- [ ] **Step 4: Verify provider tests pass**

Run:

```bash
uv run pytest tests/test_usage_hook.py -q
```

Expected: all usage hook tests pass.

### Task 4: Benchmarks and Docs

**Files:**
- Modify: `tests/test_benchmark_fixture.py`
- Modify: `scripts/compare_reranking.py`
- Modify: `scripts/evaluate_mteb_reranking.py`
- Modify: `src/ranksmith/_mteb_eval.py`
- Modify: `README.md`
- Modify: `README.ko.md`
- Modify: `docs/specs/spec_pairwise_ranking_prompting.md`
- Modify: `docs/wiki/01_decisions.md`
- Modify: `docs/wiki/02_architecture.md`

- [ ] **Step 1: Write failing benchmark tests**

Add tests for:
- fixture reaches relevant docs with `PairwiseStrategy(passes=1)`
- compare runner estimates `prp_sliding_k` calls as `2 * passes * max(document_count - 1, 0)`
- MTEB method normalization accepts `prp_sliding_k@N`

- [ ] **Step 2: Verify tests fail**

Run:

```bash
uv run pytest tests/test_benchmark_fixture.py tests/test_mteb_eval.py -q
```

Expected: tests fail because benchmark support is not implemented.

- [ ] **Step 3: Implement benchmark and docs updates**

Add:
- `prp_sliding_k` to `compare_reranking.py`
- `prp_sliding_k@N` to MTEB evaluation
- call estimates for PRP
- README usage/cost notes
- architecture/decision docs
- mark spec implementation checklist items as complete where implemented

- [ ] **Step 4: Verify targeted benchmark tests pass**

Run:

```bash
uv run pytest tests/test_benchmark_fixture.py tests/test_mteb_eval.py -q
```

Expected: tests pass.

### Task 5: Full Verification

**Files:**
- All changed files

- [ ] **Step 1: Run full verification**

Run:

```bash
./scripts/verify.sh
```

Expected: verification completes successfully.

- [ ] **Step 2: Update final spec state**

If verification passes, update `docs/specs/spec_pairwise_ranking_prompting.md`:
- mark implemented checklist items `[x]`
- set status to `Completed`

- [ ] **Step 3: Re-run verification if docs or code changed after Step 1**

Run:

```bash
./scripts/verify.sh
```

Expected: verification completes successfully.
