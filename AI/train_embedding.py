import json
import numpy as np
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 데이터 로드
with open("health_guidelines_training.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 중복 식품 제거
food_dict = {d["food"]: d for d in data}
data = list(food_dict.values())
print(f"중복 제거 후 데이터 수: {len(data)}")

# 데이터 준비
texts = [d["text"] for d in data]
labels = [d["label"] for d in data]

# 임베딩 생성
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, convert_to_numpy=True)

# 학습/테스트 분할
X_train, X_val, y_train, y_val = train_test_split(
    embeddings, labels, test_size=0.3, random_state=42, stratify=labels
)

# 분류기 학습
clf = LogisticRegression(C=0.1, max_iter=1000)
clf.fit(X_train, y_train)

# 교차 검증
scores = cross_val_score(clf, embeddings, labels, cv=5, scoring="f1_macro")
print(f"5-폴드 F1 스코어: {scores.mean():.4f} ± {scores.std():.4f}")

# 테스트 데이터로 평가
y_pred = clf.predict(X_val)
print("\n실제 라벨 (처음 10개):", y_val[:10])
print("예측 라벨 (처음 10개):", y_pred[:10])
print("\n분류기 성능 분석:")
print(classification_report(y_val, y_pred, target_names=["부정 (0)", "긍정 (1)"]))

# FAISS 벡터 저장소 생성
hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_texts(texts, hf_embeddings, metadatas=data)
vector_store.save_local("faiss_index")

# 모델 저장 (파인튜닝 생략, 필요 시 추가)
model.save("fine_tuned_mini_lm")
print("모델 및 FAISS 인덱스 저장 완료")