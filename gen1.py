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
        # 모델 변경: gemma2:2b 사용
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
        response = ollama.chat(
            model=self.phi_model,
            messages=[{"role": "user", "content": user_input}]
        )
        return response["message"]["content"]

    def resolve_ticker(self, company_name):
        prompt = f"주식 회사 '{company_name}'의 올바른 티커 심볼은 무엇입니까? 한 단어로 대답해 주세요."
        ticker = self.generate_phi_response(prompt)
        return ticker.strip().upper()

    def get_related_stocks(self, stock_ticker):
        """
        해당 주식의 섹터 정보를 조회한 후, 미리 정의한 섹터별 매핑을 통해 관련 주식을 반환합니다.
        (실제 환경에서는 보다 정교한 스크리너를 사용할 수 있음)
        """
        stock = yf.Ticker(stock_ticker)
        info = stock.info
        sector = info.get("sector")
        if not sector:
            return f"{stock_ticker}의 섹터 정보를 찾을 수 없습니다."
        # 예시 매핑 (필요에 따라 추가 확장)
        sector_mapping = {
            "Technology": ["AAPL", "MSFT", "GOOGL", "FB", "IBM"],
            "Healthcare": ["JNJ", "PFE", "MRK", "ABBV", "TMO"],
            "Financial Services": ["JPM", "BAC", "WFC", "C", "GS"],
        }
        if sector in sector_mapping:
            related = sector_mapping[sector]
            return f"{stock_ticker}의 섹터({sector})에 해당하는 관련 주식: {', '.join(related)}"
        else:
            return f"{stock_ticker}의 섹터({sector})에 해당하는 관련 주식을 찾을 수 없습니다."

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_input = update.message.text
        lower_input = user_input.lower()

        # "관련 주식" 요청: 모델 답변 대신 야후 파이낸스 정보를 사용
        if "관련 주식" in lower_input:
            # 회사 이름이나 티커를 "관련 주식" 앞에 입력했다고 가정
            company_name = user_input.split("관련 주식")[0].strip()
            if not company_name:
                company_name = user_input.strip()
            ticker = self.resolve_ticker(company_name)
            response = self.get_related_stocks(ticker)
        # "이슈 알려줘" 요청: 구글 커스텀 서치 API를 사용
        elif "이슈 알려줘" in lower_input:
            ticker = user_input.split("이슈 알려줘")[0].strip().upper()
            response = self.search_news(ticker)
        # "자세하게 알려줘" 요청: 야후 파이낸스를 사용해 상세 정보 제공
        elif "자세하게 알려줘" in lower_input:
            company_input = user_input.split("자세하게 알려줘")[0].strip()
            ticker = self.resolve_ticker(company_input)
            response = self.get_stock_info(ticker)
        # 그 외는 gemma2:2b 모델의 답변을 사용
        else:
            response = self.generate_phi_response(user_input)

        await update.message.reply_text(response)

    def run_telegram_bot(self):
        application = ApplicationBuilder().token(self.telegram_token).build()
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        application.run_polling()


if __name__ == "__main__":
    bot = StockBot()
    bot.run_telegram_bot()

