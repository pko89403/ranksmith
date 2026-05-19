# Changelog

## 0.2.0

- Add Pairwise Ranking Prompting via `PairwiseStrategy` and `AsyncPairwiseStrategy`.
- Add async pair-order parallelism for PRP without changing PRP traversal semantics.
- Keep public reranking algorithms limited to `rankgpt_sliding_window` and `prp_sliding_k`.
- Remove public `direct` and `sliding_window` algorithm options.
- Add runnable PRP example and MTEB benchmark reporting with LLM call counts.
- Update README benchmark notes with native MTEB top-20 and top-100 call accounting.
