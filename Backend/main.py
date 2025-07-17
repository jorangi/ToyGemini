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

# gemini_router를 등록합니다.
app.include_router(gemini_router)
# commands_router를 등록합니다. prefix를 제거하여 /write-file 주소를 그대로 사용하도록 합니다.
app.include_router(commands_router)

