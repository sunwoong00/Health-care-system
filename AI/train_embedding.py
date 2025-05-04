import json
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DATA_PATH = os.path.join(BASE_DIR, 'AI', 'health_guidelines_training.json')
GUIDELINES_PATH = os.path.join(BASE_DIR, 'BE', 'health_guidelines.json')
OUTPUT_MODEL_PATH = os.path.join(BASE_DIR, 'AI', 'fine_tuned_mini_lm')
FAISS_INDEX_PATH = os.path.join(BASE_DIR, 'AI', 'faiss_index')

# 1. 데이터 로드
try:
    with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"학습 데이터셋 {TRAIN_DATA_PATH}이 존재하지 않습니다.")

# 데이터프레임 변환
df = pd.DataFrame(train_data)
texts = df['text'].tolist()
labels = df['label'].tolist()

# 2. 임베딩 모델 파인튜닝
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
train_examples = [InputExample(texts=[text], label=label) for text, label in zip(texts, labels)]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.BatchHardTripletLoss(model=model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=10,
    output_path=OUTPUT_MODEL_PATH
)
print(f"임베딩 모델 파인튜닝 완료, 저장 경로: {OUTPUT_MODEL_PATH}")

# 3. 분류기 학습
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)
X_train_emb = model.encode(X_train, convert_to_tensor=True).cpu().numpy()
X_val_emb = model.encode(X_val, convert_to_tensor=True).cpu().numpy()

classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train_emb, y_train)

y_pred = classifier.predict(X_val_emb)
accuracy = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
print(f"분류기 평가 - 정확도: {accuracy:.4f}, F1 스코어: {f1:.4f}")

# 4. FAISS 검색에 파인튜닝 모델 적용
try:
    with open(GUIDELINES_PATH, 'r', encoding='utf-8') as f:
        guidelines_data = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"가이드라인 데이터셋 {GUIDELINES_PATH}이 존재하지 않습니다.")

documents = [item.get('health_benefits', '') for item in guidelines_data]
fine_tuned_embeddings = HuggingFaceEmbeddings(model_name=OUTPUT_MODEL_PATH)
vector_store = FAISS.from_texts(documents, fine_tuned_embeddings)
vector_store.save_local(FAISS_INDEX_PATH)
print(f"FAISS 벡터 저장소 저장 완료: {FAISS_INDEX_PATH}")