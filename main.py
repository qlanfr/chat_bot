import os
import ollama
import yfinance as yf
import requests
import psycopg2
import numpy as np
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

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
        self.phi_model = "gemma2:2b"

    def get_embedding(self, text):

        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        return response["embedding"]

    def find_best_match(self, user_input):

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
        """Phi-4 모델을 사용해 사용자 입력에 대한 답변을 생성합니다."""
        response = ollama.chat(
            model=self.phi_model,
            messages=[{"role": "user", "content": user_input}]
        )
        return response["message"]["content"]

    def resolve_ticker(self, company_name):

        prompt = f"주식 회사 '{company_name}'의 올바른 티커 심볼은 무엇입니까? 한 단어로 대답해 주세요."
        ticker = self.generate_phi_response(prompt)
        # 티커 문자열에 공백이 있으면 제거하고 대문자로 변환합니다.
        return ticker.strip().upper()

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        텔레그램 메시지를 수신하여 적절한 주식 정보를 반환합니다.
        
        자연어 명령어 예시:
          - "양자 컴퓨터 관련 주식 알려줘" → 관련 주식 정보 반환
          - "리게팅 컴퓨터 이슈 알려줘" → 주식 관련 최신 뉴스 반환
          - "애플 자세하게 알려줘 PBR, PER, ROE, 시가총액" → 애플의 상세 주식 정보 반환
          
        만약 사용자가 티커를 모르는 경우, phi-4 모델이 회사 이름에서 티커를 추론합니다.
        """
        user_input = update.message.text.lower()

        if "관련 주식" in user_input:
            topic = user_input.replace("관련 주식 알려줘", "").strip()
            response = self.find_best_match(topic)
            if response is None:
                response = f"'{topic}' 관련 주식을 찾을 수 없습니다."
        elif "이슈 알려줘" in user_input:
            stock_ticker = user_input.replace("이슈 알려줘", "").strip().upper()
            response = self.search_news(stock_ticker)
        elif "자세하게 알려줘" in user_input:
            # "자세하게 알려줘" 이후에 입력된 내용을 회사 이름 또는 티커로 해석
            company_input = user_input.replace("자세하게 알려줘", "").strip()
            # 먼저, phi-4를 이용해 회사 이름에서 올바른 티커를 추출합니다.
            ticker = self.resolve_ticker(company_input)
            response = self.get_stock_info(ticker)
        else:
            # 저장된 데이터에 없는 경우, Phi-4 모델을 사용하여 답변 생성
            response = self.find_best_match(user_input)
            if response is None:
                response = self.generate_phi_response(user_input)

        await update.message.reply_text(response)

    def run_telegram_bot(self):
        application = ApplicationBuilder().token(self.telegram_token).build()
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        application.run_polling()


if __name__ == "__main__":
    bot = StockBot()
    bot.run_telegram_bot()

