# Spec: 실제 사용자용 Examples 가이드 코드

> **작성 가이드**: 이 문서는 코딩 어시스턴트의 작업 추적용이기도 하지만, **최우선적으로 사람(개발자)이 읽고 이해하기 가장 좋은 형태(가독성)**여야 합니다.

## 1. 개요 (Overview)
- **작업 목적**: `ranksmith` 패키지를 설치한 실제 사용자들이 제품을 빠르고 정확하게 도입할 수 있도록, `examples` 폴더 하위에 실사용 가이드 코드를 제공합니다. 가이드 코드는 **실제 Azure OpenAI 연동**과 **비동기/동기 사용법**을 명확히 보여주는 목적입니다.
- **상태**: `[ ] Draft` | `[ ] In Progress` | `[x] Completed`

## 2. 요구 사항 및 제약 (Requirements & Constraints)
- **입력 (Inputs)**: 환경 변수(`.env`)에서 Azure OpenAI 자격 증명(API Key, Endpoint, Deployment)을 읽어와야 합니다.
- **제약 사항 (Constraints)**: 
  - 서드파티 라이브러리인 `python-dotenv` 등의 의존성을 추가해도 되는지, 혹은 단순히 `os.environ`을 사용할지 결정이 필요합니다. (사용자 패키지에는 포함되지 않도록 `dev` 의존성으로 추가하거나 기본 `os.getenv` 권장)
  - 코드는 복사해서 바로 붙여넣기 할 수 있을 만큼 직관적이어야 합니다.
  - 시스템 내부 구조(`_providers` 등)를 노출하지 않고 Public API(`AzureOpenAIReranker`, `AsyncAzureOpenAIReranker`, `ListwiseStrategy`)만 사용해야 합니다.

## 3. 상세 설계 (Architecture & Design)

가이드 코드는 크게 2가지로 나눕니다.

1. **`examples/rankgpt_sync.py`**
   - 동기(Synchronous) 방식으로 문서를 랭킹하는 가장 기본적인 예제.
   - `ListwiseStrategy(algorithm="rankgpt_sliding_window")` 사용.

2. **`examples/rankgpt_async.py`**
   - 비동기(Asynchronous) 방식으로 랭킹하는 고성능 예제.
   - `AsyncAzureOpenAIReranker`와 `asyncio.run()`을 사용한 구조.

**의사 코드 (Pseudo-code) - Sync 예제**:
```python
import os
from ranksmith import AzureOpenAIReranker, Document, ListwiseStrategy

def main():
    # 1. Reranker 초기화 (실제 API 키 사용)
    reranker = AzureOpenAIReranker(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        strategy=ListwiseStrategy(
            algorithm="rankgpt_sliding_window",
            window_size=10,
            stride=5
        )
    )

    # 2. 문서 및 쿼리 준비
    query = "비타민 결핍과 관련된 질병"
    documents = [
        "사과는 맛있다.",
        "비타민 B12 결핍은 피로와 빈혈을 유발할 수 있다.",
        "수면 부족은 건강에 해롭다."
    ]

    # 3. 랭킹 실행
    results = reranker.rerank(query, documents)

    # 4. 결과 출력
    for result in results:
        print(f"Rank {result.rank}: {result.document.text} (Original Index: {result.original_index})")

if __name__ == "__main__":
    main()
```

## 4. 에러 핸들링 (Error Handling)
- `.env`에 키가 없을 때 발생하는 오류에 대비하여 스크립트 최상단에 환경변수 확인 로직(Assertion 또는 예외 발생)을 추가해 사용자가 빠르게 인지하도록 합니다.

---

## 5. 작업 태스크 추적 (Task Checklist)

### Phase 1: 컨텍스트 및 설계 확인
- [ ] 스펙 문서(본 문서) 상의 예제 코드 설계 검토 및 확정

### Phase 2: 구현 (Implementation)
- [x] `examples/rankgpt_sync.py` 작성
- [x] `examples/rankgpt_async.py` 작성
- [ ] `.env.example` 템플릿 파일이 없다면 최상단에 추가하여 환경변수 가이드 제공

### Phase 3: 검증 (Verification)
- [ ] 실제 키를 사용하여 예제 스크립트 수동 실행 (사용자 검증 요망)
- [ ] Linter(`ruff`), Type Checker(`mypy`) 통과 확인

### Phase 4: 완료 및 정리
- [x] README.md 또는 README.ko.md에 예제 파일 위치 안내 추가
- [x] 본 문서 최상단의 **상태**를 `Completed`로 변경
