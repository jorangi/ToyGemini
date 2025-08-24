# main.py
import os
import dotenv
dotenv.load_dotenv()

import uvicorn
import logging
from google import genai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from gemini_router import router as gemini_router
from commands_router import router as commands_router
from fastapi.staticfiles import StaticFiles

class HealthCheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # 로그 메시지에 "/public/longText.txt"가 포함되어 있으면
        # False를 반환하여 해당 로그를 무시(출력 안 함)하도록 합니다.
        return "/public/longText.txt" not in record.getMessage()
    
client = genai.Client()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/public", StaticFiles(directory="../Frontend/public"), name="public")
# gemini_router를 등록합니다.
app.include_router(gemini_router)
# commands_router를 등록합니다. prefix를 제거하여 /write-file 주소를 그대로 사용하도록 합니다.
app.include_router(commands_router)

logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
