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
import json

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


def market_cap(value):
    try:
        cap = int(value)
    except (ValueError, TypeError):
        return value

    parts = []
    조 = cap // 10**12
    remainder = cap % 10**12
    if 조:
        parts.append(f"{조}조")
    억 = remainder // 10**8
    remainder %= 10**8
    if 억:
        parts.append(f"{억:,}억")
    만 = remainder // 10**4
    remainder %= 10**4
    if 만:
        parts.append(f"{만:,}만")
    if remainder:
        parts.append(f"{remainder:,}")
    return " ".join(parts) + " 달러"


class StockBot:
    def __init__(self):
        self.telegram_token = TELEGRAM_TOKEN
        self.google_api_key = GOOGLE_API_KEY
        self.google_cse_id = GOOGLE_CSE_ID
        self.model = "gemma2:2b"

    def get_embedding(self, text):
        response = ollama.embeddings(model=self.model, prompt=text)
        return response["embedding"]

    def find_best_match(self, user_input):
    
        user_embedding = self.get_embedding(user_input)
        cursor.execute("SELECT question, answer, embedding FROM stock_chatbot_data")
        rows = cursor.fetchall()
        best_similarity = float("inf")
        best_answer = None
        for question, answer, embedding in rows:
            embedding_vector = np.array(embedding)
            similarity = cosine(user_embedding, embedding_vector)
            if similarity < best_similarity:
                best_similarity = similarity
                best_answer = answer
        return best_answer

    def stock_data(self, stock_ticker):
        stock = yf.Ticker(stock_ticker)
        try:
            info = stock.info
        except Exception:
            info = {}

        pbr = info.get("priceToBook")
        per = info.get("trailingPE")
        roe = info.get("returnOnEquity")
        market_value = info.get("marketCap")

        try:
            fast_info = stock.fast_info
        except Exception:
            fast_info = {}

        if market_value is None:
            market_value = fast_info.get("marketCap", "N/A")
        if per is None:
            per = fast_info.get("trailingPE", "N/A")
        if pbr is None:
            pbr = "N/A"
        if roe is None:
            roe = "N/A"

        market_value_formatted = (
            market_cap(market_value) if market_value != "N/A" else market_value
        )

        return (
            f"📈 {stock_ticker} 정보:\n"
            f"PBR: {pbr}\n"
            f"PER: {per}\n"
            f"ROE: {roe}\n"
            f"시가총액: {market_value_formatted}"
        )

    def googl_news(self, stock_ticker):

        search_query = f"{stock_ticker} 주식 뉴스"
        url = (
            f"https://www.googleapis.com/customsearch/v1?q={search_query}"
            f"&key={self.google_api_key}&cx={self.google_cse_id}"
        )
        response = requests.get(url).json()
        results = response.get("items", [])
        news_list = []
        for item in results:
            title = item.get("title", "")
            link = item.get("link", "")
            news_list.append(f"{title}\n{link}")
        prompt = (
            "다음 뉴스 기사 목록에서, 최근 이슈이면서 중요하다고 판단되는 기사 1~2개의 링크만 "
            "선별하여 간결하게 출력해 주세요.\n\n" + "\n\n".join(news_list)
        )
        filtered = self.ge_answer(prompt)
        analysis_prompt = (
            "위에서 선별된 뉴스 기사들을 바탕으로, 해당 이슈들이 주식 시장에 미치는 영향과 주요 포인트에 대해 "
            "간략하게 분석해 주세요."
        )
        analysis = self.ge_answer(analysis_prompt)
        final_response = f"{filtered}\n\n[추가 분석]\n{analysis}"
        return final_response if final_response.strip() else "관련 뉴스가 없습니다."

    def ge_answer(self, user_input):

        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": user_input}],
        )
        return response["message"]["content"]

    def find_tk(self, company_name):

        lower_name = company_name.lower()
        if "구글" in lower_name or "google" in lower_name:
            return "GOOGL"
        if "애플" in lower_name or "apple" in lower_name:
            return "AAPL"
        prompt = (
            f"다음 회사의 주식 티커 심볼을 알려주세요. 가능한 경우 표준 티커 심볼만 대문자 한 단어로 출력해 주세요:\n"
            f"회사명: {company_name}"
        )
        ticker = self.ge_answer(prompt)
        return ticker.strip().upper()

    def ai_answer(self, user_input):

        prompt = (
            f"사용자의 주식 관련 질문을 아래 유형 중 하나로 분류해 주세요.\n"
            f"1. 주식 상세 정보 요청 (특정 회사의 상세 정보 요청)\n"
            f"2. 주식 뉴스 요청\n"
            f"3. 관련 주식 추천 요청\n"
            f"4. 일반 대화\n"
            f"질문: {user_input}\n"
            f"답변은 1, 2, 3, 4 중 하나의 숫자만 출력해 주세요."
        )
        result = self.ge_answer(prompt)
        try:
            intent = int(result.strip())
        except:
            intent = 4
        return intent

    async def sand_mg(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            user_input = update.message.text.strip()
            intent = self.ai_answer(user_input)
            db_answer = self.find_best_match(user_input)
            if db_answer:
                response = db_answer
            else:
                if intent == 1:
                    ticker = self.find_tk(user_input)
                    response = self.stock_data(ticker)
                elif intent == 2:
                    ticker = self.find_tk(user_input)
                    response = self.googl_news(ticker)
                elif intent == 3:
                    response = self.ge_answer(user_input)
                else:
                    response = self.ge_answer(user_input)
            await update.message.reply_text(response)
        except Exception as e:
            await update.message.reply_text("죄송합니다 다시 한번 질문 해주세요.")

    def run_bot(self):
        application = ApplicationBuilder().token(self.telegram_token).build()
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.sand_mg))
        application.run_polling()


if __name__ == "__main__":
    bot = StockBot()
    bot.run_bot()

