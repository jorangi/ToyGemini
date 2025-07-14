# backend/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import os
import dotenv
from fastapi.middleware.cors import CORSMiddleware



dotenv.load_dotenv()  # .env에서 API 키 로드

# Gemini API 키 설정
genai.configure(api_key=os.getenv("AIzaSyAIkMbZ30JfJm1GcHeBu4YzyA08E50b1mo"))

# 모델 인스턴스 생성
model = genai.GenerativeModel("gemini-2.5-flash")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite 기본 포트
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatInput(BaseModel):
    message: str

@app.post("/gemini/message")
async def chat_with_gemini(data: ChatInput):
    try:
        response = model.generate_content(data.message)
        return {"response": response.text}
    except Exception as e:
        return {"error": str(e)}
