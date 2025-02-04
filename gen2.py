import os
import re
import ollama
import yfinance as yf
import requests
import psycopg2
import numpy as np
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

# 환경변수 로드 (.env 파일)
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

# PostgreSQL 데이터베이스 연결 (실제 사용 시 예외 처리 및 연결 종료 관리 필요)
conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)
cursor = conn.cursor()


class StockBot:
    def __init__(self):
        self.telegram_token = TELEGRAM_TOKEN
        self.google_api_key = GOOGLE_API_KEY
        self.google_cse_id = GOOGLE_CSE_ID
        # 모델 변경: phi4에서 gemma2:2b로 변경
        self.phi_model = "gemma2:2b"

    def get_embedding(self, text):
        """
        ollama의 'nomic-embed-text' 모델을 사용하여 텍스트 임베딩을 가져옵니다.
        """
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        return response["embedding"]

    def find_best_match(self, user_input):
        """
        DB에 저장된 질문 및 임베딩 데이터를 기반으로 사용자 입력과 가장 유사한 질문을 찾고
        해당하는 답변을 반환합니다.
        """
        user_embedding = self.get_embedding(user_input)
        cursor.execute("SELECT question, answer, embedding FROM stock_chatbot_data")
        rows = cursor.fetchall()

        best_match = None
        best_similarity = float("inf")
        best_answer = ""

        for question, answer, embedding in rows:
            embedding_vector = np.array(embedding)
            similarity = cosine(user_embedding, embedding_vector)
            if similarity < best_similarity:
                best_similarity = similarity
                best_match = question
                best_answer = answer

        return best_answer if best_match else None

    def get_stock_info(self, stock_ticker):
        """
        yfinance를 이용해 주어진 티커의 주식 정보를 조회합니다.
        """
        stock = yf.Ticker(stock_ticker)
        info = stock.info
        pbr = info.get("priceToBook", "N/A")
        per = info.get("trailingPE", "N/A")
        roe = info.get("returnOnEquity", "N/A")
        market_cap = info.get("marketCap", "N/A")

        return (f"📈 {stock_ticker} 정보:\n"
                f"PBR: {pbr}\n"
                f"PER: {per}\n"
                f"ROE: {roe}\n"
                f"시가총액: {market_cap}")

    def search_news(self, stock_ticker):
        """
        구글 커스텀 서치 API를 사용해 주식 관련 최신 뉴스를 검색합니다.
        """
        search_query = f"{stock_ticker} stock news"
        url = (f"https://www.googleapis.com/customsearch/v1?q={search_query}"
               f"&key={self.google_api_key}&cx={self.google_cse_id}")

        response = requests.get(url).json()
        results = response.get("items", [])

        news_list = []
        for item in results[:3]:
            title = item["title"]
            link = item["link"]
            news_list.append(f"{title}\n{link}")

        return "\n\n".join(news_list) if news_list else "관련 뉴스를 찾을 수 없습니다."

    def generate_phi_response(self, user_input):
        """
        gemma2:2b 모델을 사용하여 사용자 입력에 대해 응답을 생성합니다.
        """
        response = ollama.chat(
            model=self.phi_model,
            messages=[{"role": "user", "content": user_input}]
        )
        return response["message"]["content"]

    def resolve_ticker(self, company_name):
        """
        회사 이름으로부터 올바른 티커를 추출합니다.
        gemma2:2b 모델에 '한 단어로 대답해 주세요'라는 조건을 부여하여 티커만 추출하도록 요청합니다.
        """
        prompt = f"주식 회사 '{company_name}'의 올바른 티커 심볼은 무엇입니까? 한 단어로 대답해 주세요."
        ticker = self.generate_phi_response(prompt)
        return ticker.strip().upper()

    def classify_query(self, user_input):
        """
        gemma2:2b 모델을 이용하여 사용자 질문을 아래 세 가지 카테고리로 분류합니다.
          - '관련 주식'
          - '이슈 알려줘'
          - '자세하게 알려줘'

        추가 주의 사항: 파싱 및 분류 오류를 보다 견고하게 처리하기 위해 regex를 사용합니다.
        예상되는 출력 형식이 아닐 경우 모든 플래그를 False로 설정합니다.
        """
        prompt = (
            "다음 문장이 '관련 주식', '이슈 알려줘', '자세하게 알려줘' 중 어느 카테고리에 해당하는지 판단해 주세요. "
            "각 카테고리에 대해 'True' 또는 'False'로 대답해 주세요. "
            "예: 관련 주식: True, 이슈: False, 자세한: False\n"
            f"문장: {user_input}"
        )
        classification = self.generate_phi_response(prompt)
        flags = {"관련 주식": False, "이슈": False, "자세한": False}

        try:
            for category in flags:
                pattern = rf"{category}:\s*(True|False)"
                match = re.search(pattern, classification, re.IGNORECASE)
                if match:
                    flags[category] = match.group(1).strip().lower() == "true"
                else:
                    flags[category] = False
        except Exception as e:
            print("분류 결과 파싱 오류:", e)
            flags = {"관련 주식": False, "이슈": False, "자세한": False}
        return flags

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        텔레그램 메시지를 수신하여 사용자 질문의 카테고리에 따라 적절한 주식 정보를 반환합니다.
        질문이 들어오면 먼저 "응답중입니다"라는 메시지를 보내고 나머지 처리를 진행합니다.
        
        자연어 명령어 예시:
          - "테슬라 관련 주식 알고 싶어" → 테슬라와 관련된 주식을 찾아서 알려줍니다.
          - "리게팅 컴퓨터 이슈 알려줘" → 주식 관련 최신 뉴스 반환
          - "애플 자세하게 알려줘 PBR, PER, ROE, 시가총액" → 애플의 상세 주식 정보 반환
        """
        user_input = update.message.text

        # 먼저 "응답중입니다" 메시지를 즉시 전송
        await update.message.reply_text("응답중입니다")

        # gemma2:2b 모델을 사용해 질문의 유형을 분류합니다.
        flags = self.classify_query(user_input)

        if flags.get("관련 주식"):
            # "관련 주식" 기능: 입력에서 "관련 주식" 앞에 있는 키워드를 추출
            if "관련 주식" in user_input:
                parts = user_input.split("관련 주식")
                topic = parts[0].strip()
            else:
                topic = user_input.strip()
            if not topic:
                topic = user_input.strip()
            response = self.find_best_match(topic)
            if response is None:
                # fallback: gemma2:2b 모델을 이용하여 관련 주식 정보를 생성
                fallback_prompt = f"{topic}와 관련된 주식을 리스트 형식으로 주요 주식들을 알려줘."
                response = self.generate_phi_response(fallback_prompt)
        elif flags.get("이슈"):
            # 이슈 관련 분기: "이슈 알려줘"라는 문구 제거 후 티커 추출
            stock_ticker = user_input.replace("이슈 알려줘", "").strip().upper()
            response = self.search_news(stock_ticker)
        elif flags.get("자세한"):
            # 자세한 정보 분기: "자세하게 알려줘"라는 문구 제거 후 회사 이름 추출 및 티커 해결
            company_input = user_input.replace("자세하게 알려줘", "").strip()
            ticker = self.resolve_ticker(company_input)
            response = self.get_stock_info(ticker)
        else:
            response = self.generate_phi_response(user_input)

        await update.message.reply_text(response)

    def run_telegram_bot(self):
        """
        텔레그램 봇을 초기화하고 메시지 핸들러를 등록하여 폴링 방식으로 동작시킵니다.
        """
        application = ApplicationBuilder().token(self.telegram_token).build()
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        application.run_polling()


if __name__ == "__main__":
    bot = StockBot()
    bot.run_telegram_bot()

