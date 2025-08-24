from fastapi import APIRouter, Request, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from google import genai
from google.genai import types
from metatools.handlers import run_optimization_workflow
import re, json, json5, datetime, asyncio, os, dotenv, inspect, uuid, chromadb
from prompt_builder import build_prompt, extract_json_from_text
from commands_router import execute_shell, CommandRequest
from sql_router import get_db_schema, Conversation, User, Alias, Session as SessionModel, SessionLocal, ToolCallLog
from collections import defaultdict, deque
from sqlalchemy.orm import Session
from pydantic import BaseModel
from fastapi import Depends
from sqlalchemy import text, update
from tools.basic_definitions import basic_tool_definitions
from tools.basic_handlers import basic_action_handlers
from llm_utils import call_gemini_agent
from agent import Agent
from config import tool_manager, sync_tools_with_vector_db, tool_collection, embedding_model






def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class ChatPayload(BaseModel):
    conversation_id: str
    speaker_id: int
    message_content: str

MESSAGE_CACHE_SIZE = 30
recent_messages = defaultdict(lambda: deque(maxlen=MESSAGE_CACHE_SIZE))

dotenv.load_dotenv()

MODEL_PRIORITY_LIST = os.getenv("GEMINI_MODEL_PRIORITY_LIST").split(',')
print(MODEL_PRIORITY_LIST)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 확인해주세요.")
client = genai.Client() 

router = APIRouter()

from collections.abc import Mapping
from typing import Any
def validate_sql_against_schema(sql: str, schema: Mapping[str, Any]) -> tuple[bool, str]:
    """
    SQL 쿼리에 사용된 테이블과 컬럼이 실제 DB 스키마에 존재하는지 검사합니다.
    """
    if not isinstance(schema, dict):
        return False, "스키마 정보가 유효하지 않습니다."

    sql_lower = sql.lower().strip()
    
    # INSERT INTO 구문은 컬럼 검사에서 예외 처리 (VALUES 내용에 키워드가 많기 때문)
    if sql_lower.startswith("insert"):
        return True, "INSERT 쿼리는 유효성 검사를 건너뜁니다."

    # 테이블 이름 추출
    table_matches = re.findall(r"(?:from|update|into|join)\s+`?(\w+)`?", sql_lower)
    if not table_matches:
        return True, "SQL에서 테이블 이름을 찾을 수 없어 검사를 건너뜁니다 (예: CREATE TABLE)."
    
    table_name = table_matches[0]
    
    if table_name not in schema:
        return False, f"테이블 '{table_name}'이(가) 존재하지 않습니다. 사용 가능한 테이블: {list(schema.keys())}"
        
    available_columns = {col['name'].lower() for col in schema.get(table_name, [])}
    
    # 사용된 컬럼 이름 추출
    used_columns = set()
    select_match = re.search(r"select\s+(.*?)\s+from", sql_lower)
    if select_match and select_match.group(1).strip() != '*':
        used_columns.update(col.strip().strip('`') for col in select_match.group(1).split(','))
        
    general_matches = re.findall(r"(\w+)\s*(?:=|like|in|>|<)", sql_lower)
    used_columns.update(general_matches)
    
    for col in used_columns:
        if col not in available_columns:
            return False, f"컬럼 '{col}'이(가) '{table_name}' 테이블에 존재하지 않습니다. 사용 가능한 컬럼: {list(available_columns)}"
            
    return True, "유효성 검사 통과"


print("✅ 모든 도구와 핸들러가 성공적으로 조립되었습니다.")
print(f"   - 총 {len(tool_manager.action_handlers)}개의 핸들러 로드됨")
print(f"   - 총 {len(tool_manager.all_definitions)}개의 도구 정의 로드됨")




@router.post("/gemini_stream")
async def gemini_stream(request: Request, background_tasks: BackgroundTasks):
    try:
        data = await request.json()
        user_input = data.get("prompt", "")
        session_id = data.get("session_id") or str(uuid.uuid4())

        agent_instance = Agent(
        user_goal=user_input,
        session_id=session_id,
        background_tasks=background_tasks,
        model_priority_list=MODEL_PRIORITY_LIST
    )
        
    except Exception:
        return JSONResponse({"error": "Invalid request body"}, status_code=400)
    
    return StreamingResponse(agent_instance.stream_run(), media_type="text/event-stream")
# ----- 특정 키워드 포함 메시지 검색 -----
def find_message(conversation_id: str, keyword: str, db: Session):
    # 캐시에서 먼저 찾기
    if conversation_id in recent_messages:
        for item in reversed(recent_messages[conversation_id]):
            if keyword in item["message_content"]:
                return item
    # 없으면 DB에서 최신 메시지 검색
    row = db.query(Conversation)\
        .filter(
            Conversation.conversation_id == conversation_id,
            Conversation.message_content.like(f"%{keyword}%")
        )\
        .order_by(Conversation.timestamp.desc())\
        .first()
    if row:
        return {
            "speaker_id": row.speaker_id,
            "message_content": row.message_content,
            "timestamp": row.timestamp,
        }
    return None
