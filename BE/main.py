from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, db
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.docstore.document import Document
import datetime  # timestamp 생성용

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    raise

try:
    json_path = os.path.join(os.path.dirname(__file__), "health_guidelines.json")
    with open(json_path, "r", encoding="utf-8") as f:
        guidelines = json.load(f)
    
    documents = [Document(page_content=item["guideline"], metadata={"category": item["category"]}) for item in guidelines]
    if not documents:
        raise ValueError("No guidelines found in JSON")

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(split_docs, embeddings)
    print(f"RAG initialized successfully with {len(split_docs)} document chunks")
except Exception as e:
    print(f"RAG initialization failed: {str(e)}")
    raise

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set. Please set it in BE/.env.")
genai.configure(api_key=GOOGLE_API_KEY)
print("Gemini API configured successfully")

def query_gemini(prompt: str):
    print("Querying Gemini API (Flash)...")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 1500,
                "temperature": 0.3,
                "top_p": 0.9
            }
        )
        print("Gemini API response received")
        return response.text
    except Exception as e:
        print(f"Gemini API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Gemini API query failed: {str(e)}")

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
    timestamp: str = None

class FoodAnalysisRequest(BaseModel):
    user: Login
    food: Food

@app.post("/login")
async def login_user(data: Login):
    try:
        ref = db.reference(f"users/{data.name}")
        ref.set(data.dict())
        print(f"User saved: {data.name}")
        return {"message": "User data saved successfully", "user_id": data.name}
    except Exception as e:
        print(f"Login error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving data: {str(e)}")

@app.get("/check_user")
async def check_user(name: str):
    try:
        user_ref = db.reference(f'users/{name}')
        user_data = user_ref.get()
        print(f"Check user {name}: exists={bool(user_data)}")
        return {"exists": bool(user_data)}
    except Exception as e:
        print(f"Check user error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error checking user: {str(e)}")

@app.post("/add_meal")
async def add_meal(data: dict):
    try:
        print(f"Received meal data: {json.dumps(data, ensure_ascii=False)}")
        user = data.get("user")
        meal = data.get("meal")
        
        if not user or not meal:
            raise HTTPException(status_code=400, detail="Invalid request data")
        
        user_ref = db.reference(f"users/{user['name']}/meals")
        existing_meals = user_ref.get() or []
        print(f"Existing meals: {json.dumps(existing_meals, ensure_ascii=False)} (type: {type(existing_meals)})")
        
        if not isinstance(existing_meals, list):
            print("Resetting invalid meals data to empty list")
            existing_meals = []
        
        existing_meals.append({
            "breakfast": meal["breakfast"],
            "lunch": meal["lunch"],
            "dinner": meal["dinner"],
            "timestamp": meal.get("timestamp") or datetime.datetime.now().isoformat()
        })
        user_ref.set(existing_meals)
        print(f"Updated meals: {json.dumps(existing_meals, ensure_ascii=False)}")
        
        return {"message": "Meal data saved successfully"}
    except Exception as e:
        print(f"Add meal error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving meal data: {str(e)}")

@app.get("/get_meals")
async def get_meals(name: str):
    try:
        print(f"Fetching meals for user: {name}")
        user_ref = db.reference(f"users/{name}/meals")
        meals = user_ref.get()
        print(f"Raw meals data: {json.dumps(meals, ensure_ascii=False)} (type: {type(meals)})")
        if meals is None:
            print("No meals found, returning empty list")
            return {"meals": []}
        if not isinstance(meals, list):
            print(f"Invalid meals type: {type(meals)}, resetting to empty list")
            return {"meals": []}
        print("Returning meals as list")
        return {"meals": meals}
    except Exception as e:
        print(f"Get meals error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching meal data: {str(e)}")

@app.post("/food")
async def analyze_food(data: FoodAnalysisRequest):
    print(f"Received food analysis request: {json.dumps(data.dict(), ensure_ascii=False)}")
    try:
        if not all([data.food.breakfast, data.food.lunch, data.food.dinner]):
            raise HTTPException(status_code=400, detail="All meal fields are required")

        if not data.user.name or data.user.age <= 0 or not data.user.gender or data.user.height <= 0 or data.user.weight <= 0:
            raise HTTPException(status_code=400, detail="Invalid user data")

        user = data.user
        # 저장 제거: /add_meal에서 이미 저장됨
        user_ref = db.reference(f'users/{user.name}/meals')
        existing_meals = user_ref.get() or []
        print(f"Existing meals: {json.dumps(existing_meals, ensure_ascii=False)} (type: {type(existing_meals)})")
        
        if not isinstance(existing_meals, list):
            print("Resetting invalid meals data to empty list")
            existing_meals = []
        
        meal_history_str = "\n".join(
            [f"{i+1}일차: 아침({m['breakfast']}), 점심({m['lunch']}), 저녁({m['dinner']})"
             for i, m in enumerate(existing_meals)]
        )

        query = (
            f"사용자: {user.name}, 나이: {user.age}세, 성별: {'남성' if user.gender == 'male' else '여성'}, "
            f"키: {user.height}cm, 몸무게: {user.weight}kg\n"
            f"식사 기록:\n{meal_history_str}\n"
            f"이 정보를 바탕으로 식습관 분석과 건강 추천을 제공하세요."
        )
        try:
            retrieved_docs = vector_store.similarity_search(query, k=3)
            retrieved_texts = [doc.page_content for doc in retrieved_docs]
            print(f"Retrieved documents: {json.dumps(retrieved_texts, ensure_ascii=False)}")
        except Exception as rag_e:
            print(f"RAG error: {str(rag_e)}")
            retrieved_texts = []

        prompt = f"""
        당신은 전문 영양사이자 건강 관리사입니다. 사용자의 프로필, 식단 데이터, 그리고 참고 정보를 바탕으로 상세하고 정확한 식단 분석을 제공하세요.
        ### 참고 정보
        {'(없음)' if not retrieved_texts else chr(10).join(retrieved_texts)}
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
        다음을 한국어로 작성하세요. **반드시 아래 형식을 정확히 준수**하며, 다른 스타일(예: **, 숫자, 자유 텍스트)을 사용하지 마세요:
        - 각 항목은 '###'으로 시작하는 소제목으로 구분.
        - 각 항목의 내용은 '- '로 시작하는 목록으로 작성.
        - 각 섹션은 최소 2개 이상의 목록 항목 포함.
        - 모든 섹션에 내용이 있어야 하며, 빈 섹션은 허용되지 않음.
        1. ### 영양 균형
           - 사용자의 나이, 성별, 키, 몸무게, BMI를 고려하여 식단의 영양소 균형(탄수화물, 단백질, 지방, 비타민 등)을 평가.
           - 부족하거나 과다한 영양소를 구체적으로 설명.
        2. ### 개선점
           - 식단을 더 건강하게 만들기 위한 대체 식품, 섭취량 조절, 조리법 제안.
        3. ### 건강 위험
           - 식단과 신체 정보를 바탕으로 잠재적 건강 위험(예: 고혈압, 당뇨, 비만) 분석 및 예방법.
        4. ### 맞춤 추천
           - 사용자의 프로필에 맞춘 1일 식단 예시.
        """
        analysis = query_gemini(prompt)
        print(f"Gemini response: {json.dumps(analysis, ensure_ascii=False)}")
        
        required_sections = ['### 영양 균형', '### 개선점', '### 건강 위험', '### 맞춤 추천']
        retries = 0
        max_retries = 2
        while retries < max_retries:
            if all(section in analysis for section in required_sections):
                break
            print(f"Gemini response missing sections, retry {retries+1}/{max_retries}")
            analysis = query_gemini(prompt)
            retries += 1
        
        if not all(section in analysis for section in required_sections):
            print("Gemini response still invalid, using fallback")
            analysis = "\n".join([
                "### 영양 균형\n- 식단 데이터 부족으로 기본 평가 제공.\n- 균형 잡힌 식사를 위해 단백질과 채소 섭취 필요.",
                "### 개선점\n- 채소 섭취를 늘리세요.\n- 가공식품을 줄이세요.",
                "### 건강 위험\n- 불균형 식단으로 영양 부족 위험.\n- 정기적인 건강 검진 추천.",
                "### 맞춤 추천\n- 아침: 오트밀과 과일.\n- 점심: 샐러드와 닭가슴살."
            ])

        return {
            "analysis": analysis,
            "documents": retrieved_texts
        }
    except Exception as e:
        print(f"Food analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing food habits: {str(e)}")