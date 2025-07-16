# main.py
import os
import dotenv
from google import genai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from gemini_router import router as gemini_router
from commands_router import router as commands_router
dotenv.load_dotenv()

client = genai.Client()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 스트리밍 라우터 등록
app.include_router(gemini_router)
app.include_router(commands_router, prefix="/cmd", tags=["CommandShell"])