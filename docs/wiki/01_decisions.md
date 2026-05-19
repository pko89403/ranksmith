# 결정 기록

## D001 Package 이름
Decision: `ranksmith`

Status: accepted

Reason: 기억하기 쉽고 ranking과 관련 있으며, 프로젝트 시작 시점에 사용 가능했다.

## D002 주요 사용자 경험
Decision: class-first API through `AzureOpenAIReranker`.

Status: accepted

Reason: provider 설정과 reranking 실행이 분리되어 명확하다.

## D003 문서 입력
Decision: accept `Sequence[str | Document]`.

Status: accepted

Reason: 문자열은 첫 사용을 쉽게 만들고, `Document`는 `id`와 `metadata`를 보존한다.

## D004 결과 계약
Decision: `RerankResult(document, rank, original_index, metadata)`.

Status: accepted

Reason: score를 강제하지 않으면서 향후 listwise, pointwise, pairwise 전략을 모두 수용한다.

## D005 Error 정책
Decision: fast fail.

Status: accepted

Reason: reranking 품질이 조용히 왜곡되면 안 된다.

## D006 Strategy 모델
Decision: separate strategy from algorithm.

Status: accepted

Reason: 향후 pointwise, pairwise, tournament, bayesian, confidence algorithm을 수용한다.

## D007 Reference 통제
Decision: if references are insufficient, stop and ask the user.

Status: accepted

Reason: 사용자가 scope, 조사 출처, 구현 방향을 통제한다.

## D008 Pairwise PRP public API
Decision: add `PairwiseStrategy` and `AsyncPairwiseStrategy`.

Status: accepted

Reason: PRP는 listwise permutation이 아니라 pairwise comparison을 비교 단위로 사용하므로, `ListwiseStrategy`에 억지로 넣으면 provider 계약과 알고리즘 의미가 흐려진다.

## D009 Pairwise provider contract
Decision: keep `LLMProvider.rank()` for listwise ranking and add pairwise `compare()`.

Status: accepted

Reason: 기존 JSON permutation 계약을 유지하면서 PRP의 binary choice prompt와 strict winner parsing을 분리한다.
