from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import os
import dotenv
import json, ast
from google import genai
from google.genai import types
import asyncio
from google.api_core import exceptions
import re, json5, httpx

from prompt_builder import build_prompt, extract_json_from_text
from commands_router import execute_shell, CommandRequest
from sql_router import execute_raw_sql, get_db_schema, get_db_schema_for_tables

dotenv.load_dotenv()

MODEL_PRIORITY_LIST = os.getenv("GEMINI_MODEL_PRIORITY_LIST").split(',')
print(MODEL_PRIORITY_LIST)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 확인해주세요.")
client = genai.Client()

router = APIRouter()

tools = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="db_schema_query",
                description="현재 데이터베이스의 모든 테이블 이름과 각 테이블의 상세 스키마(컬럼 정보)를 조회합니다. 특정 테이블의 스키마만 필요할 경우 'get_specific_table_schema'를 사용하는 것이 더 효율적입니다.",
            ),
            # [추가] 새로운 도구 'get_specific_table_schema' 정의
            types.FunctionDeclaration(
                name="get_specific_table_schema",
                description="테이블 이름 목록을 받아서, 해당하는 테이블들의 스키마 정보만 반환합니다.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "table_names": types.Schema(
                            type=types.Type.ARRAY,
                            items=types.Schema(type=types.Type.STRING),
                            description="스키마를 조회할 테이블 이름들의 리스트. 예: ['characters', 'items']"
                        ),
                        "thought": types.Schema(type=types.Type.STRING, description="이 테이블들을 선택한 이유.")
                    },
                    required=["table_names", "thought"]
                )
            ),
            types.FunctionDeclaration(
                name="execute_sql_query",
                description="주어진 SQL 쿼리 문자열을 데이터베이스에 직접 실행합니다.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "sql_query": types.Schema(type=types.Type.STRING, description="실행할 전체 SQL 쿼리 문장"),
                        "is_write_operation": types.Schema(type=types.Type.BOOLEAN, description="쿼리가 데이터를 변경하는 작업인 경우 true"),
                        "thought": types.Schema(type=types.Type.STRING, description="이 쿼리를 실행하기로 결정한 이유나 생각.")
                    },
                    required=["sql_query", "is_write_operation", "thought"]
                )
            ),
            types.FunctionDeclaration(
                name="execute_shell_command",
                description="운영체제 쉘(Shell)에 CMD 명령어를 실행합니다.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "command": types.Schema(type=types.Type.STRING, description="실행할 CMD 명령어."),
                        "thought": types.Schema(type=types.Type.STRING, description="이 명령어를 실행하기로 결정한 이유나 생각.")
                    },
                    required=["command", "thought"]
                )
            ),
            types.FunctionDeclaration(
                name="final_response",
                description="모든 작업이 완료되어 사용자에게 최종 답변을 할 때 사용합니다.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "answer": types.Schema(type=types.Type.STRING, description="사용자에게 전달할 최종 답변 메시지."),
                        "thought": types.Schema(type=types.Type.STRING, description="최종 답변을 하기로 결정한 이유나 생각 요약.")
                    },
                    required=["answer", "thought"]
                )
            ),
        ]
    )
]

# ... call_gemini_agent 함수는 이전과 동일 ...
# gemini_router.py 파일에 적용할 내용입니다.

# gemini_router.py 파일에 적용할 내용입니다.

def safe_json_parse(s):
    try:
        return json5.loads(s)
    except Exception as e:
        print(f"[json5 파싱 실패] {e}")
        return None

async def call_gemini_agent(
    prompt_content: list[types.Content],
    use_tools: bool = True,
    temperature: float = 0.2,
    tools=None
) -> types.GenerateContentResponse:
    
    model_priority_env = os.getenv("GEMINI_MODEL_PRIORITY_LIST")
    if not model_priority_env:
        raise RuntimeError("GEMINI_MODEL_PRIORITY_LIST 환경변수가 설정되지 않았습니다.")
    MODEL_PRIORITY_LIST = [m.strip() for m in model_priority_env.split(',')]
    
    last_exception = None
    for model_name in MODEL_PRIORITY_LIST:
        print(f"[🔄 모델 시도] '{model_name}' 모델로 API 호출을 시도합니다.")
        
        try:
            config_args = {
                "temperature": temperature,
                "safety_settings": [
                    types.SafetySetting(category='HARM_CATEGORY_HATE_SPEECH', threshold='BLOCK_NONE'),
                    types.SafetySetting(category='HARM_CATEGORY_HARASSMENT', threshold='BLOCK_NONE'),
                    types.SafetySetting(category='HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold='BLOCK_NONE'),
                    types.SafetySetting(category='HARM_CATEGORY_DANGEROUS_CONTENT', threshold='BLOCK_NONE'),
                ]
            }
            if use_tools and tools is not None:
                config_args['tools'] = tools

            response = await client.aio.models.generate_content(
                    model=model_name, 
                    contents=prompt_content,
                    config=types.GenerateContentConfig(**config_args)
                )
            
            print(f"[✅ 모델 성공] '{model_name}' 모델 호출에 성공했습니다.")
            return response

        except (exceptions.ResourceExhausted, exceptions.TooManyRequests, exceptions.ServiceUnavailable) as e:
            print(f"[⚠️ 모델 실패] '{model_name}' 모델의 할당량(429)이 소진되었습니다. 다음 모델을 시도합니다. 예외: {type(e)}, {e}")
            last_exception = e
            continue # 다음 모델로 계속

        except Exception as e:
            print(f"[❌ API 오류] '{model_name}' 모델에서 예상치 못한 오류가 발생했습니다: {type(e)}, {e}")
            last_exception = e
            continue  # <-- break 대신 continue (다음 모델까지 시도)

    print("[❌ 모든 모델 실패] 사용 가능한 모든 모델의 호출에 실패했습니다.")

    error_function_call = types.FunctionCall(
        name='final_response',
        args={
            'answer': f"모든 API 모델 호출에 실패하여 요청을 처리할 수 없습니다. 잠시 후 다시 시도해주세요. (최종 오류: {str(last_exception)})",
            'thought': "모든 우선순위의 모델에서 API 호출이 실패했습니다."
        }
    )
    
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    parts=[types.Part(function_call=error_function_call)]
                )
            )
        ]
    )


# [✨ 추가] SQL 유효성 검사 (가드레일) 함수
# [✨ 버그 수정] SQL 유효성 검사 (가드레일) 함수
def validate_sql_against_schema(sql: str, schema: dict) -> (bool, str):
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

# --- 액션 핸들러 정의 ---
async def handle_write_file(action_input):
    filename = action_input.get("filename")
    content = action_input.get("content")
    if not filename or content is None:
        return {"error": "filename/content is missing"}

    # commands_router.py의 API로 요청
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8000/write-file",  # 실제 서버 주소/포트로 맞추기!
                json={
                    "filepath": filename,  # 반드시 API 스펙과 맞춰야 함!
                    "content": content
                },
                timeout=5
            )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API 오류 {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": f"write-file API 호출 실패: {e}"}
    
async def handle_db_schema_query(action_input):
    db_schema_response = await asyncio.to_thread(get_db_schema)
    if db_schema_response["status"] == "success":
        return {"db_schema": db_schema_response["schema"]}
    else:
        raise Exception(f"DB 스키마 조회 실패: {db_schema_response['message']}")

# [추가] 새로운 도구를 처리할 핸들러 함수
async def handle_get_specific_table_schema(action_input):
    table_names = action_input.get("table_names")
    if not table_names or not isinstance(table_names, list):
        raise ValueError("'table_names'는 리스트 형태여야 합니다.")
    
    db_schema_response = await asyncio.to_thread(get_db_schema_for_tables, table_names)
    if db_schema_response["status"] == "success":
        return {"db_schema": db_schema_response["schema"]}
    else:
        raise Exception(f"특정 테이블 스키마 조회 실패: {db_schema_response['message']}")

async def handle_execute_sql_query(action_input):
    # ... (기존과 동일)
    sql_query = action_input.get("sql_query")
    is_write_op = action_input.get("is_write_operation", False)
    if not sql_query: raise ValueError("'sql_query'가 누락되었습니다.")
    
    sql_exec_result = await asyncio.to_thread(execute_raw_sql, sql_query, is_write_op)
    if sql_exec_result["status"] == "success":
        return {"sql_execution_result": sql_exec_result}
    else:
        raise Exception(f"SQL 실행 실패: {sql_exec_result['message']}")

async def handle_execute_shell_command(action_input):
    # ... (기존과 동일)
    command = action_input.get("command")
    if not command: raise ValueError("'command'가 누락되었습니다.")
    
    shell_result = await asyncio.to_thread(execute_shell, CommandRequest(command=command))
    if shell_result is None:
        raise Exception("쉘 실행 결과가 None입니다. execute_shell 함수를 확인해주세요.")

    if shell_result.get("exit_code") == 0:
        stdout_str = shell_result.get('stdout', '출력 없음').strip()
        stderr_str = shell_result.get('stderr', '에러 없음').strip()
        return f"쉘 명령어 실행 결과입니다.\n[STDOUT]:\n{stdout_str}\n\n[STDERR]:\n{stderr_str}"
    else:
        raise Exception(f"쉘 실행 실패: STDOUT='{shell_result.get('stdout')}', STDERR='{shell_result.get('stderr')}'")


# [수정] action_handlers에 새로운 핸들러 등록
action_handlers = {
    "write_file": handle_write_file,
    "db_schema_query": handle_db_schema_query,
    "get_specific_table_schema": handle_get_specific_table_schema, # <-- 추가
    "execute_sql_query": handle_execute_sql_query,
    "execute_shell_command": handle_execute_shell_command,
}

# ... _process_agent_request 및 메인 API 엔드포인트는 이전과 동일 ...
# [✨ 수정] 핵심 에이전트 루프 (모든 로직 통합)
MAX_AGENT_ITERATIONS = 20
async def _process_agent_request(user_goal: str):
    final_answer = None
    response_logs = []
    
    initial_prompt_text = build_prompt(f"**사용자 요청 (최종 목표):**\n\"{user_goal}\"")
    print(initial_prompt_text)
    prompt_content = [types.Content(role="user", parts=[types.Part(text=initial_prompt_text)])]

    try:
        current_schema_response = await asyncio.to_thread(get_db_schema)
        current_schema = current_schema_response.get("schema", {})
    except Exception as e:
        final_answer = f"DB 스키마 로딩 실패: {e}"
        return {"status": "failed", "response": final_answer, "logs": [final_answer]}

    for iteration in range(MAX_AGENT_ITERATIONS):
        print(f"\n======== Agent Iteration {iteration + 1} ========")
        response_logs.append(f"\n======== Agent Iteration {iteration + 1} ========")
        
        response = await call_gemini_agent(prompt_content, use_tools=True, temperature=0.2)
        
        try:
            candidate = response.candidates[0]
            parts = getattr(candidate.content, "parts", [])
            
            if not parts:
                final_answer = "모델 응답에 parts가 없습니다."
                response_logs.append(final_answer)
                break

            part = parts[0]
            function_call = getattr(part, "function_call", None)
            
            if not function_call:
                raw_text = getattr(part, "text", None)
                if raw_text:
                    json_str = extract_json_from_text(raw_text)
                    print("====[DEBUG] json_str 끝부분====")
                    print(json_str[-100:])  # 마지막 100자
                    print("============================")
                    print("len(json_str):", len(json_str))
                    if json_str:
                        json_obj = safe_json_parse(json_str)
                        if json_obj and "action" in json_obj and "action_input" in json_obj:
                            action = json_obj["action"]
                            action_input = json_obj.get("action_input", {})
                            thought = json_obj.get("thought", "")
                            is_final_answer = json_obj.get("is_final_answer", False)
                            # 아래 function_call 분기 코드로 이어서 진행
                        else:
                            final_answer = f"[JSON 구조 아님] {raw_text}"
                            response_logs.append(final_answer)
                            print(f"[⚠️ JSON 구조 아님] {final_answer}")
                            break
                    else:
                        final_answer = raw_text
                        response_logs.append(f"[Function Call/텍스트] {final_answer}")
                        print(f"[⚠️ Function Call/텍스트] {final_answer}")
                        break
                else:
                    final_answer = "Function Call/텍스트 응답이 모두 없습니다."
                    response_logs.append(final_answer)
                    break
            else:
                # [원래 function_call 분기] ↓
                action = getattr(function_call, "name", None)
                action_input = dict(getattr(function_call, "args", {})) if hasattr(function_call, "args") else {}
                thought = action_input.pop('thought', '생각 없음')

            # [function_call 및 JSON fallback 모두 여기로 진입]
            try:
                print(f"[✅ Agent Response] Thought: {thought.encode('utf-8', 'replace').decode('utf-8')}")
                print(f"  Action: {action}, Input: {action_input}")
            except Exception as e:
                print(f"[⚠️ Print Error] Agent Response 출력 중 오류: {e}")
            response_logs.append(f"Thought: {thought}\nAction: {action}, Input: {json.dumps(action_input, ensure_ascii=False)}")

            if action == "final_response":
                final_answer = action_input.get("answer", "최종 답변이 명시되지 않았습니다.")
                try:
                    safe_to_print = final_answer.encode('utf-8', 'replace').decode('utf-8')
                    print(f"[🏁 Final Answer] {safe_to_print}")
                except Exception as e:
                    print(f"[⚠️ Print Error] Final Answer 출력 중 오류: {e}")
                response_logs.append(f"Final Answer: {final_answer}")
                break

            observation_result = None

            # 기존 액션 핸들링
            if action == "execute_sql_query":
                sql_query = action_input.get("sql_query", "")
                is_valid, error_msg = validate_sql_against_schema(sql_query, current_schema)
                if not is_valid:
                    observation_result = {"error": f"SQL 유효성 검사 실패: {error_msg}"}
                else:
                    observation_result = await handle_execute_sql_query(action_input)
            elif action in action_handlers:
                observation_result = await action_handlers[action](action_input)
            else:
                raise ValueError(f"알 수 없는 Agent 액션: '{action}'.")

            try:
                print(f"[✅ Observation] {json.dumps(observation_result, ensure_ascii=False, indent=2)}")
            except Exception as e:
                print(f"[⚠️ Print Error] Observation 출력 중 오류 발생: {e}")
            response_logs.append(f"Observation: {json.dumps(observation_result, ensure_ascii=False)}")

            # 스키마 갱신
            if action == "execute_sql_query" and action_input.get("is_write_operation", False):
                sql_lower = action_input.get("sql_query","").lower()
                if "create" in sql_lower or "alter" in sql_lower or "drop" in sql_lower:
                    print("[🔄 스키마 캐시 업데이트]")
                    current_schema_response = await asyncio.to_thread(get_db_schema)
                    current_schema = current_schema_response.get("schema", {})

            # 다음 프롬프트에 observation 추가
            prompt_content.append(candidate.content)
            prompt_content.append(types.Content(
                role="user",
                parts=[types.Part(
                    function_response=types.FunctionResponse(
                        name=action,
                        response={"result": observation_result}
                    )
                )]
            ))

        except Exception as e:
            import traceback
            traceback.print_exc()
            final_answer = f"Agent 루프 오류: {e}"
            response_logs.append(f"ERROR: {final_answer}")
            break

    if final_answer is None:
        final_answer = "최대 반복 횟수에 도달했지만 최종 답변을 얻지 못했습니다."
        response_logs.append(final_answer)
    
    return {
        "status": "success" if "오류" not in final_answer and "실패" not in final_answer else "failed",
        "response": final_answer,
        "logs": response_logs
    }
# [✨ 수정] 메인 API 엔드포인트
@router.post("/gemini")
async def gemini(request: Request):
    """
    모든 요청을 단일화된 AI 에이전트(_process_agent_request)로 보내 처리합니다.
    """
    try:
        data = await request.json()
        user_input = data.get("prompt", "")

        if not user_input:
            return JSONResponse({"error": "Prompt is empty"}, status_code=400)

        # '문지기' 없이 모든 요청을 바로 에이전트에게 전달
        execution_result = await _process_agent_request(user_input)
        
        if execution_result["status"] == "success":
            return JSONResponse({"response": execution_result["response"], "logs": execution_result["logs"]})
        else:
            return JSONResponse({"response": execution_result["response"], "logs": execution_result["logs"]}, status_code=500)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[❌ 전역 에러] {e}")
        return JSONResponse({"error": str(e)}, status_code=500)