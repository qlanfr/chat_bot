### 📌 StockBot - 주식 정보 챗봇

StockBot은 텔레그램을 통해 실시간 주식 정보를 제공하는 AI 기반 챗봇입니다. 주식 시가총액, 뉴스, 관련 주식 등 다양한 기능을 제공합니다.

-------

🚀 주요 기능

✅ 주식 정보 조회: 주가, PBR, PER, ROE, 시가총액 등을 확인할 수 있습니다.


✅ 뉴스 검색 및 분석: 구글 뉴스 API를 이용해 최신 주식 관련 뉴스를 검색하고, AI가 분석하여 주요 포인트를 제공합니다.

✅ 관련 주식 추천: 특정 주식과 관련된 추천 주식을 제공합니다.

✅ AI 기반 대화: AI가 사용자의 질문을 분석하여 적절한 응답을 생성합니다.

------

🛠️ 설치 방법

1️⃣ 필수 패키지 설치

pip install -r requirements.txt

2️⃣ 환경 변수 설정

.env 파일을 생성하여 아래 내용을 추가하세요.

TELEGRAM_TOKEN=YOUR_TELEGRAM_BOT_TOKEN

GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY

GOOGLE_CSE_ID=YOUR_GOOGLE_CSE_ID

DB_NAME=your_database

DB_USER=your_db_user

DB_PASSWORD=your_db_password

DB_HOST=localhost

DB_PORT=5432

▶ 실행 방법

python gen3.py

------------------

🔍 사용 예시

텔레그램에서 아래와 같은 질문을 입력하면 해당 정보를 제공합니다:

애플의 시가 총액 알려줘

테슬라 관련 뉴스 알려줘

구글과 관련된 주식 추천해줘


---------------


🔧 향후 개선 사항

📌 AI 모델 파인 튜닝 예정: 현재는 기본 모델을 사용 중이며, 추후 파인 튜닝을 통해 정확도를 개선할 예정입니다.

📌 추가 데이터 학습: 실시간 주식 시장 데이터를 반영하여 더욱 정밀한 응답을 제공하도록 개선할 계획입니다.

📌 멀티모델 지원: 필요에 따라 다양한 AI 모델을 활용하여 질문 유형별로 최적의 응답을 제공할 예정입니다.


-----------------



🤖 사용한 AI 모델

Gemma2:2B (주식 관련 질문 분석 및 답변 생성)

-----------------


📜 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.


----------

# 이미지

![사용](https://github.com/qlanfr/chat_bot/blob/master/chat_bot.png)


