# Reranking Smoke Fixture Sources

This fixture contains a small excerpted sample from SciFact for offline reranking smoke tests.

## Source
- Dataset: SciFact
- Paper: "Fact or Fiction: Verifying Scientific Claims" (Wadden et al., 2020)
- Upstream repository: https://github.com/allenai/scifact
- BEIR mirror: https://huggingface.co/datasets/BeIR/scifact
- Qrels mirror: https://huggingface.co/datasets/BeIR/scifact-qrels
- License: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/

## Changes
- Selected three development-set claims.
- Selected a small set of candidate documents for each claim.
- Reformatted claims, document excerpts, and qrels into JSONL.
- Kept only document titles and the first two abstract sentences to keep the fixture small.

## Scope
This fixture is for deterministic regression and smoke testing only. It is not large enough for statistically meaningful benchmark claims or published-score comparison.
