import json

# 기본 가이드
base_guidelines = [
    {"guideline": "고혈압 환자는 저염식 식단을 권장합니다. 하루 나트륨 섭취량은 2,300mg 이하로 제한하세요.", "category": "고혈압"},
    {"guideline": "당뇨병 환자는 저탄수화물 식단과 규칙적인 운동이 필요합니다. 정제된 설탕 섭취를 피하세요.", "category": "당뇨"},
    {"guideline": "체중 감량을 위해 칼로리 섭취를 조절하세요. 하루 500kcal 감소로 주 0.5kg 감량 가능합니다.", "category": "체중 관리"},
    {"guideline": "심장 건강을 위해 오메가-3 지방산이 풍부한 생선을 섭취하세요. 연어, 고등어, 정어리가 좋습니다.", "category": "심장 건강"},
    {"guideline": "비타민 D 부족 시 연어, 정어리, 강화 우유를 섭취하세요. 햇볕 쬐기도 도움이 됩니다.", "category": "영양소"},
    {"guideline": "철분 부족은 빈혈을 유발할 수 있습니다. 시금치, 붉은 고기, 렌틸콩을 식단에 추가하세요.", "category": "영양소"},
    {"guideline": "칼슘은 뼈 건강에 필수적입니다. 유제품, 브로콜리, 아몬드를 섭취하세요.", "category": "영양소"},
    {"guideline": "고지혈증 환자는 트랜스지방과 포화지방 섭취를 줄이고, 올리브 오일을 사용하세요.", "category": "고지혈증"},
    {"guideline": "식이섬유는 소화를 돕고 혈당을 안정시킵니다. 귀리, 사과, 당근을 추천합니다.", "category": "소화 건강"},
    {"guideline": "수분 섭취는 신진대사를 돕습니다. 하루 2리터 이상의 물을 마시세요.", "category": "일반 건강"},
]

# 카테고리별 추가 가이드
categories = ["고혈압", "당뇨", "체중 관리", "심장 건강", "영양소", "고지혈증", "소화 건강", "일반 건강"]
guidelines = base_guidelines.copy()

for cat in categories:
    for i in range(1, 13):
        if cat == "고혈압":
            guideline = f"{cat} 환자는 {['저염 된장', '무염 버터', '샐러드 드레싱', '해조류', '칼륨 풍부 과일'][i%5]}을 활용한 식단을 고려하세요."
        elif cat == "당뇨":
            guideline = f"{cat} 관리에는 {['현미밥', '통밀빵', '퀴노아', '저당 과일', '녹색 잎 채소'][i%5]} 섭취가 도움이 됩니다."
        elif cat == "체중 관리":
            guideline = f"{cat}를 위해 {['고단백 저칼로리 식품', '채소 위주 식단', '간헐적 단식', '저지방 유제품', '견과류 소량'][i%5]}을 추천합니다."
        elif cat == "심장 건강":
            guideline = f"{cat}에 {['오메가-3 보충제', '아보카도', '올리브 오일', '베리류', '녹차'][i%5]}가 유익합니다."
        elif cat == "영양소":
            guideline = f"{['비타민 C', '비타민 E', '마그네슘', '오메가-3', '프로바이오틱스'][i%5]} 보충을 위해 {['오렌지', '아몬드', '바나나', '연어', '요거트'][i%5]}를 섭취하세요."
        elif cat == "고지혈증":
            guideline = f"{cat} 환자는 {['콩류', '통곡물', '채소 스무디', '저지방 단백질', '견과류'][i%5]}를 식단에 추가하세요."
        elif cat == "소화 건강":
            guideline = f"{cat}을 위해 {['발효 식품', '식이섬유', '생강차', '프로바이오틱스', '수분'][i%5]} 섭취를 늘리세요."
        else:  # 일반 건강
            guideline = f"{cat} 유지를 위해 {['균형 잡힌 식단', '규칙적인 운동', '충분한 수면', '스트레스 관리', '정기 건강검진'][i%5]}을 실천하세요."
        guidelines.append({"guideline": guideline, "category": cat})

# 100개로 조정
guidelines = guidelines[:100]

# JSON 저장
with open("health_guidelines.json", "w", encoding="utf-8") as f:
    json.dump(guidelines, f, ensure_ascii=False, indent=2)
print(f"Generated JSON with {len(guidelines)} guidelines")
