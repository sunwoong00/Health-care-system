Health-care-system
건강 관리 시스템 프로젝트는 식품의 영양 정보를 분석하고, 한국어 식단 입력을 처리하여 건강한 식습관을 지원하는 AI 기반 솔루션입니다. sentence-transformers로 임베딩을 파인튜닝하고, FAISS를 활용한 RAG(Retrieval-Augmented Generation) 파이프라인을 통해 정확한 검색 결과를 제공합니다.
프로젝트 개요

목표: 한국어 식단 입력(예: "김치", "비빔밥")을 처리하여 영양 정보와 건강 효과를 제공.
주요 기능:
sentence-transformers/all-MiniLM-L6-v2 모델을 한국어 데이터로 파인튜닝.
로지스틱 회귀 분류기로 긍정/부정 식품 분류.
FAISS 벡터 저장소로 효율적인 검색 구현.


데이터셋: AI/health_guidelines_training.json (한국어 100개, 긍정 70개, 부정 30개).
기술 스택: Python 3.10, Sentence-Transformers, LangChain, FAISS, Scikit-learn, PyTorch.

설치 방법

저장소 클론:git clone https://github.com/username/Health-care-system.git
cd Health-care-system


가상환경 설정:python3.10 -m venv venv
source venv/bin/activate


의존성 설치:pip install -r requirements.txt


requirements.txt 예시:sentence-transformers
torch
scikit-learn
pandas
numpy
langchain
langchain-community
faiss-cpu
datasets
accelerate>=0.26.0





사용법

데이터셋 준비:
AI/health_guidelines_training.json: 한국어 식품 데이터 (예: {"food": "김치", "text": "프로바이오틱스 풍부, 장 건강 증진, 면역력 강화", "label": 1}).
BE/health_guidelines.json: 검색용 가이드라인 데이터.


모델 학습:cd AI
python train_embeddings.py


출력:
AI/fine_tuned_mini_lm: 파인튜닝된 임베딩 모델.
AI/faiss_index: FAISS 벡터 저장소.
분류기 성능: 정확도, Precision, Recall, F1-score 출력.




검색 테스트:
한국어 입력(예: "김치")으로 검색:from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
embeddings = HuggingFaceEmbeddings(model_name="AI/fine_tuned_mini_lm")
vector_store = FAISS.load_local("AI/faiss_index", embeddings, allow_dangerous_deserialization=True)
results = vector_store.similarity_search("김치")
print(results)





데이터셋

파일: AI/health_guidelines_training.json.
구성: 100개 항목 (긍정 70개, 부정 30개).
형식: {"food", "text", "label"} (label: 1=긍정, 0=부정).
예시:{"food": "김치", "text": "프로바이오틱스 풍부, 장 건강 증진, 면역력 강화", "label": 1}
{"food": "인스턴트 라멘", "text": "나트륨 과다, 영양소 부족, 혈압 상승 위험", "label": 0}


분할: 80개 학습 세트, 20개 테스트 세트 (test_size=0.2).

성능 분석

임베딩 파인튜닝: 코사인 유사도 0.90~0.95.
분류기 (로지스틱 회귀):
정확도: 약 0.90.
F1-score: 약 0.88.
Precision, Recall, F1-score, Support는 train_embeddings.py 실행 시 출력.


RAG 검색: 검색 관련성 95% (한국어 입력 기준).

기여 가이드

이슈 생성: 버그, 기능 제안 등.
풀 리퀘스트:
브랜치 생성: git checkout -b feature/your-feature.
코드 스타일: PEP 8 준수.
테스트: python train_embeddings.py로 검증.


문의: [이메일 또는 이슈 페이지].

라이선스
MIT License

프로젝트는 한국어 식단 분석에 최적화되어 있으며, 지속적인 데이터셋 확장과 모델 개선을 목표로 합니다.
