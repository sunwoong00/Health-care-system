from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, db
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dotenv import load_dotenv
import os

# .env 파일 로드 (BE 폴더 기준)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# .env 로드 디버깅
print("Current working directory:", os.getcwd())
print(".env path:", os.path.join(os.path.dirname(__file__), '.env'))
print("GOOGLE_API_KEY from env:", os.getenv("GOOGLE_API_KEY"))

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
try:
    cred = credentials.Certificate("health-6743a-firebase-adminsdk-fbsvc-9c274b8cbb.json")
    if not firebase_admin._apps:
        firebase_admin.initialize_app(
            cred,
            {"databaseURL": "https://health-6743a-default-rtdb.firebaseio.com/"}
        )
    print("Firebase initialized successfully")
except Exception as e:
    print(f"Firebase initialization failed: {str(e)}")

# RAG 설정
try:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    documents = [
        "고혈압 환자는 저염식 식단을 권장합니다.",
        "당뇨병 환자는 저탄수화물 식단과 규칙적인 운동이 필요합니다.",
        "체중 감량을 위해 칼로리 섭취를 조절하세요.",
        "심장 건강을 위해 오메가-3 지방산이 풍부한 생선을 섭취하세요."
    ]
    doc_embeddings = embedder.encode(documents)
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(doc_embeddings)
    print("RAG initialized successfully")
except Exception as e:
    print(f"RAG initialization failed: {str(e)}")

# Gemini API 설정
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set. Please set it in BE/.env.")
genai.configure(api_key=GOOGLE_API_KEY)
print("Gemini API configured successfully")

# Gemini API 호출 함수
def query_gemini(prompt: str):
    print("Querying Gemini API (Flash)...")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 1500,
                "temperature": 0.6,
                "top_p": 0.9
            }
        )
        print("Gemini API response received")
        return response.text
    except Exception as e:
        print(f"Gemini API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Gemini API query failed: {str(e)}")

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

class FoodAnalysisRequest(BaseModel):
    user: Login
    food: Food

@app.post("/login")
async def login_user(data: Login):
    try:
        ref = db.reference("users")
        new_user_ref = ref.push()
        new_user_ref.set(data.dict())
        return {"message": "User data saved successfully", "user_id": new_user_ref.key}
    except Exception as e:
        print(f"Login error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving data: {str(e)}")

@app.get("/check_user")
async def check_user(name: str):
    try:
        user_ref = db.reference(f'users/{name}').get()
        return {"exists": bool(user_ref)}
    except Exception as e:
        print(f"Check user error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error checking user: {str(e)}")

@app.post("/add_meal")
async def add_meal(data: dict):
    try:
        print(f"Received meal data: {data}")
        user = data["user"]
        meal = data["meal"]
        
        if not user or not meal:
            raise HTTPException(status_code=400, detail="잘못된 요청 데이터입니다.")
        
        user_ref = db.reference(f"users/{user['name']}/meals")
        new_meal_ref = user_ref.push()
        new_meal_ref.set(meal)
        
        return {"message": "Meal data saved successfully"}
    except Exception as e:
        print(f"Add meal error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving meal data: {str(e)}")

@app.get("/get_meals")
async def get_meals(name: str):
    try:
        user_ref = db.reference(f"users/{name}/meals")
        meals = user_ref.get() or {}
        return {"meals": list(meals.values())}
    except Exception as e:
        print(f"Get meals error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching meal data: {str(e)}")

@app.post("/food")
async def analyze_food(data: FoodAnalysisRequest):
    print(f"Received food analysis request: {data.dict()}")
    try:
        # 데이터 유효성 검사
        if not all([data.food.breakfast, data.food.lunch, data.food.dinner]):
            raise HTTPException(status_code=400, detail="All meal fields are required")

        # 사용자 정보 유효성
        if not data.user.name or data.user.age <= 0 or not data.user.gender or data.user.height <= 0 or data.user.weight <= 0:
            raise HTTPException(status_code=400, detail="Invalid user data")

        user = data.user
        new_food = data.food

        # Firebase에서 기존 식사 기록 가져오기
        user_ref = db.reference(f'users/{user.name}/meals')
        existing_meals = user_ref.get()

        if existing_meals is None:
            existing_meals = []
        elif isinstance(existing_meals, dict):
            existing_meals = list(existing_meals.values())

        existing_meals.append({
            "breakfast": new_food.breakfast,
            "lunch": new_food.lunch,
            "dinner": new_food.dinner
        })

        user_ref.set(existing_meals)

        meal_history_str = "\n".join(
            [f"{i+1}일차: 아침({m['breakfast']}), 점심({m['lunch']}), 저녁({m['dinner']})"
             for i, m in enumerate(existing_meals)]
        )

        # RAG: 쿼리 생성 및 문서 검색
        query = (
            f"사용자: {user.name}, 나이: {user.age}세, 성별: {'남성' if user.gender == 'male' else '여성'}, "
            f"키: {user.height}cm, 몸무게: {user.weight}kg\n"
            f"식사 기록:\n{meal_history_str}\n"
            f"이 정보를 바탕으로 식습관 분석과 건강 추천을 제공하세요."
        )
        try:
            query_embedding = embedder.encode([query])[0]
            D, I = index.search(np.array([query_embedding]), k=3)
            retrieved_docs = [documents[i] for i in I[0]]
            print(f"Retrieved documents: {retrieved_docs}")
        except Exception as rag_e:
            print(f"RAG error: {str(rag_e)}")
            retrieved_docs = []

        # Gemini API 프롬프트
        prompt = f"""
        당신은 전문 영양사이자 건강 관리사입니다. 사용자의 프로필, 식단 데이터, 그리고 참고 정보를 바탕으로 상세하고 정확한 식단 분석을 제공하세요.

        ### 참고 정보
        {'(없음)' if not retrieved_docs else chr(10).join(retrieved_docs)}

        ### 사용자 정보
        - 이름: {user.name}
        - 나이: {user.age}세
        - 성별: {'남성' if user.gender == 'male' else '여성'}
        - 키: {user.height}cm
        - 몸무게: {user.weight}kg
        - BMI: {(user.weight / ((user.height / 100) ** 2)):.1f} (참고용)

        ### 사용자 식단 기록
        {meal_history_str}

        ### 분석 요청
        다음을 한국어로 상세히 작성하세요:
        1. **영양 균형**: 사용자의 나이, 성별, 키, 몸무게, BMI를 고려하여 식단의 영양소 균형(탄수화물, 단백질, 지방, 비타민 등)을 평가하세요. 어떤 영양소가 부족하거나 과다한지 구체적으로 설명.
        2. **개선점**: 현재 식단을 더 건강하게 만들기 위해 구체적인 대체 식품, 섭취량 조절, 조리법 등을 제안하세요.
        3. **건강 위험**: 식단과 신체 정보를 바탕으로 잠재적인 건강 위험(예: 고혈압, 당뇨, 비만)을 분석하고 예방법을 제안하세요.
        4. **맞춤 추천**: 사용자의 프로필에 맞춘 1일 식단 예시를 제공하세요.

        ### 출력 형식
        - 각 항목은 소제목(###)으로 구분.
        - 간결하고 명확한 문장 사용.
        - 목록(-)을 활용해 가독성 높임.
        """

        # Gemini API 호출
        analysis = query_gemini(prompt)
        print("Analysis completed")

        return {
            "analysis": analysis,
            "documents": retrieved_docs
        }
    except Exception as e:
        print(f"Food analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing food habits: {str(e)}")