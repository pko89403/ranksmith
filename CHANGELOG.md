# Changelog

## 0.6.0

- Add `CBDRStrategy` (confidence-gain reranking): skips context reranking when
  `Conf(Q)` is already high, otherwise ranks documents by
  `Conf(Q+C) - Conf(Q)` using trained query-only and query+context
  answerability confidence scorers.
- Add `AnswerConfidenceRerankStrategy` / `AsyncAnswerConfidenceRerankStrategy`
  (experimental): ranks documents by the structural confidence of a
  per-document answer.
- Add a local LM Studio confidence pipeline (`LMStudioModelProvider`,
  `ProviderAnswerGenerator`) and CLIs to generate, train, and report on
  `Conf(Q)` / `Conf(Q+C)` scorer datasets without Azure access.
- Add README Benchmarking rows for `cbdr` and `answer_confidence`, both
  measured with out-of-domain scorers (TriviaQA and SQuAD v1.1
  respectively) and reported as measured — below the BM25 baseline here,
  not tuned to win.

## 0.5.1

- Add `SetwiseStrategy` and `AsyncSetwiseStrategy` with `setwise_heapsort`.
- Add setwise heapsort example, tests, benchmark support, and README benchmark evidence.
- Document the setwise reference mapping and implementation constraints.

## 0.3.2

- Streamline the PyPI README for clearer package discovery.
- Move detailed benchmark reproduction notes into dedicated benchmark docs.
- Use GitHub absolute links so PyPI project description links resolve correctly.
- Add PyPI project links for documentation and benchmark notes.

## 0.2.0

- Add Pairwise Ranking Prompting via `PairwiseStrategy` and `AsyncPairwiseStrategy`.
- Add async pair-order parallelism for PRP without changing PRP traversal semantics.
- Keep public reranking algorithms limited to `rankgpt_sliding_window` and `prp_sliding_k`.
- Remove public `direct` and `sliding_window` algorithm options.
- Add runnable PRP example and MTEB benchmark reporting with LLM call counts.
- Update README benchmark notes with native MTEB top-20 and top-100 call accounting.
