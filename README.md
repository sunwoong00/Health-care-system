# Health-care-system

한국어 식단 입력을 처리하여 식품의 영양 정보와 건강 효과를 제공하는 AI 기반 건강 관리 시스템입니다. `sentence-transformers`로 임베딩을 파인튜닝하고, FAISS를 활용한 RAG 파이프라인으로 정확한 검색 결과를 제공합니다.

## 프로젝트 개요

- **목표**: 한국어 식단(예: 김치, 비빔밥)을 분석하여 영양 정보 제공.
- **주요 기능**:
  - `all-MiniLM-L6-v2` 모델을 한국어 데이터로 파인튜닝.
  - 로지스틱 회귀 분류기로 긍정/부정 식품 분류.
  - FAISS 벡터 저장소로 효율적인 검색.
- **데이터셋**: `AI/health_guidelines_training.json` (한국어 100개, 긍정 70개, 부정 30개).
- **기술 스택**: Python 3.10, Sentence-Transformers, LangChain, FAISS, Scikit-learn, PyTorch.

## 설치 방법

### 1. 저장소 클론
```bash
git clone https://github.com/username/Health-care-system.git
cd Health-care-system
```

### 2. 가상환경 설정
```bash
python3.10 -m venv venv
source venv/bin/activate
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

`requirements.txt` 예시:
```
sentence-transformers
torch
scikit-learn
pandas
numpy
langchain
langchain-community
faiss-cpu
datasets
accelerate>=0.26.0
```

## 사용법

### 1. 데이터셋 준비
- **파일**:
  - `AI/health_guidelines_training.json`: 한국어 식품 데이터.
  - `BE/health_guidelines.json`: 검색용 가이드라인.
- **예시**:
  ```json
  {"food": "김치", "text": "프로바이오틱스 풍부, 장 건강 증진, 면역력 강화", "label": 1}
  {"food": "인스턴트 라멘", "text": "나트륨 과다, 영양소 부족, 혈압 상승 위험", "label": 0}
  ```

### 2. 모델 학습
```bash
cd AI
python train_embeddings.py
```

**출력**:
- `AI/fine_tuned_mini_lm`: 파인튜닝된 모델.
- `AI/faiss_index`: FAISS 벡터 저장소.
- 성능 지표: 정확도, Precision, Recall, F1-score, Support.

### 3. 검색 테스트
```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
embeddings = HuggingFaceEmbeddings(model_name="AI/fine_tuned_mini_lm")
vector_store = FAISS.load_local("AI/faiss_index", embeddings, allow_dangerous_deserialization=True)
results = vector_store.similarity_search("김치")
print(results)
```

## 데이터셋

- **파일**: `AI/health_guidelines_training.json`.
- **구성**: 100개 항목 (긍정 70, 부정 30).
- **형식**: `{"food", "text", "label"}` (1=긍정, 0=부정).
- **분할**: 학습 80개, 테스트 20개 (`test_size=0.2`).
- **특징**: 한식(김치, 비빔밥), 간식(떡볶이), 가공식품(인스턴트 라멘).

## 기여 가이드

- **이슈**: 버그, 기능 제안.
- **풀 리퀘스트**:
  1. 브랜치 생성: `git checkout -b feature/your-feature`.
  2. 코드 스타일: PEP 8.
  3. 테스트: `python AI/train_embeddings.py`.
- **문의**: GitHub 이슈 페이지.

## 라이선스

MIT License

---

한국어 식단 분석에 최적화된 AI 솔루션으로, 데이터셋 확장과 성능 개선을 목표로 합니다.