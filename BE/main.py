from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, db
from fastapi.middleware.cors import CORSMiddleware
import requests
import json

# FastAPI 앱 생성
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Firebase 초기화
cred = credentials.Certificate("health-6743a-firebase-adminsdk-fbsvc-9c274b8cbb.json")  # 서비스 계정 키 경로
if not firebase_admin._apps:
    firebase_admin.initialize_app(
        cred,
        {"databaseURL": "https://health-6743a-default-rtdb.firebaseio.com/"}
    )


# 데이터 모델 정의
class Login(BaseModel):
    name: str
    age: int
    gender: str
    height: int
    weight: int

class Food(BaseModel):
    breakfast: str
    lunch: str
    dinner: str


# Ollama 서버 주소
OLLAMA_URL = "http://localhost:11434/api/generate"

# Ollama와 통신 함수
def query_ollama(prompt: str):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2:latest", "prompt": prompt},
            stream=True  # 스트리밍 요청
        )
        response.raise_for_status()
        
        # 스트리밍 데이터를 처리
        result = ""
        for chunk in response.iter_lines():
            if chunk:
                chunk_data = chunk.decode("utf-8")
                try:
                    # JSON 데이터로 변환 후 "response" 필드 추출
                    chunk_json = json.loads(chunk_data)
                    result += chunk_json.get("response", "")
                except json.JSONDecodeError:
                    continue
        
        #print(result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama query failed: {str(e)}")


# 테스트
@app.get("/")
async def print_hello():
    return "Hello World"

# 로그인
@app.post("/login")
async def login_user(data: Login):
    try:
        ref = db.reference("users")  # users라는 경로에 데이터 저장
        new_user_ref = ref.push()
        new_user_ref.set(data.dict())
        return {"message": "User data saved successfully", "user_id": new_user_ref.key}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving data: {str(e)}")

# 하루 식습관 분석
@app.post("/food")
async def analyze_food(data: Food):
    try:
        # 1. 식습관 데이터를 기반으로 프롬프트 생성
        prompt = (
            f"당신은 최고의 건강 관리사입니다다.\n"
            f"제공되는 하루 식단을 참고하여 식단 분석을 진행하세요.\n"
            f"아침: {data.breakfast}\n"
            f"점심: {data.lunch}\n"
            f"저녁: {data.dinner}\n"
            f"다음 형식을 참고하여 식단에 대한 분석을 제공하세요요:\n"
            f"1. 영양 균형: <전체적인 영양 균형에 대한 간단한 분석을 제공하세요.>\n"
            f"2. 잠재적 개선점: <더 건강한 식단을 위해 구체적인 개선점을 제안하세요.>\n"
            f"3. 건강 위험: <이 식단과 관련된 잠재적인 건강 위험을 식별하세요.>\n"
            f"마지막으로, 전체 응답을 한국어로 번역하세요.\n"
            f"지정된 형식으로만 답변하세요."
        )

        # 2. Ollama 모델에 요청
        analysis = query_ollama(prompt)

        # 3. 분석 결과 반환
        return {"analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing food habits: {str(e)}")
