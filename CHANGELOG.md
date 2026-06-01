# Changelog

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
