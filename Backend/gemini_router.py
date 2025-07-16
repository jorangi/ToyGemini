from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import os
import dotenv
import json
from google import genai
from google.genai import types
import asyncio
from google.api_core import exceptions

from prompt_builder import classify_prompt_type, build_prompt
from commands_router import execute_shell, CommandRequest
# [수정] sql_router에서 새로운 함수 임포트
from sql_router import execute_raw_sql, create_db_tables, get_db_schema, get_db_schema_for_tables

dotenv.load_dotenv()

MODEL_PRIORITY_LIST = os.getenv("GEMINI_MODEL_PRIORITY_LIST").split(',')
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
async def call_gemini_agent(
    prompt_content: list[types.Content],
    use_tools: bool = True,
    temperature: float = 0.2
) -> types.GenerateContentResponse:
    
    last_exception = None
    for model_name in MODEL_PRIORITY_LIST:
        model_name = model_name.strip()
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
            if use_tools:
                config_args['tools'] = tools

            response = await client.aio.models.generate_content(
                    model=model_name, 
                    contents=prompt_content,
                    config=types.GenerateContentConfig(**config_args)
                )
            
            print(f"[✅ 모델 성공] '{model_name}' 모델 호출에 성공했습니다.")
            return response

        except exceptions.ResourceExhausted as e:
            print(f"[⚠️ 모델 실패] '{model_name}' 모델의 할당량이 소진되었습니다. 다음 모델을 시도합니다. (오류: {e.message})")
            last_exception = e
            continue

        except Exception as e:
            print(f"[❌ API 오류] '{model_name}' 모델에서 예상치 못한 오류가 발생했습니다: {e}")
            last_exception = e
            break

    print("[❌ 모든 모델 실패] 사용 가능한 모든 모델의 호출에 실패했습니다.")
    return types.GenerateContentResponse.from_response(
        dict(candidates=[dict(content=dict(parts=[dict(
            function_call=dict(
                name='final_response',
                args=dict(
                    answer=f"모든 API 모델 호출에 실패하여 요청을 처리할 수 없습니다. 잠시 후 다시 시도해주세요. (최종 오류: {str(last_exception)})",
                    thought="모든 우선순위의 모델에서 API 호출이 실패했습니다."
                )
            )
        )]))])
    )


# --- 액션 핸들러 정의 ---
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
    "db_schema_query": handle_db_schema_query,
    "get_specific_table_schema": handle_get_specific_table_schema, # <-- 추가
    "execute_sql_query": handle_execute_sql_query,
    "execute_shell_command": handle_execute_shell_command,
}

# ... _process_agent_request 및 메인 API 엔드포인트는 이전과 동일 ...
MAX_AGENT_ITERATIONS = 20
async def _process_agent_request(user_goal: str):
    # ... (내용 동일) ...
    final_answer = None
    response_logs = []
    prompt_content = [
        types.Content(
            role="user", 
            parts=[types.Part(text=build_prompt(user_goal, mode="system"))]
        )
    ]
    for iteration in range(MAX_AGENT_ITERATIONS):
        print(f"\n======== Agent Iteration {iteration + 1} ========")
        response_logs.append(f"\n======== Agent Iteration {iteration + 1} ========")
        response = await call_gemini_agent(prompt_content, use_tools=True, temperature=0.2)
        try:
            message = response.candidates[0].content
            if not message.parts or not hasattr(message.parts[0], "function_call") or not message.parts[0].function_call.name:
                final_answer = message.parts[0].text if message.parts and hasattr(message.parts[0], "text") else "모델이 함수를 호출하지 않고 응답을 종료했습니다."
                break
            
            function_call = message.parts[0].function_call
            action = function_call.name
            action_input = {k: v for k, v in function_call.args.items()}
            thought = action_input.pop('thought', '생각 없음')

            print(f"[✅ Agent Response] Thought: {thought}")
            print(f"  Action: {action}, Input: {action_input}")
            response_logs.append(f"Thought: {thought}\nAction: {action}, Input: {action_input}")

            if action == "final_response":
                final_answer = action_input.get("answer", "최종 답변이 명시되지 않았습니다.")
                print(f"[🏁 Final Answer] {final_answer}")
                response_logs.append(f"Final Answer: {final_answer}")
                break

            if action in action_handlers:
                observation_result = await action_handlers[action](action_input)
            else:
                raise ValueError(f"알 수 없는 Agent 액션: '{action}'.")

            print(f"[✅ Observation] {json.dumps(observation_result, ensure_ascii=False, indent=2)}")
            response_logs.append(f"Observation: {json.dumps(observation_result, ensure_ascii=False)}")
            
            prompt_content.append(message)
            prompt_content.append(
                types.Content(
                    role="user", 
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(name=action, response={"result": observation_result})
                        )
                    ]
                )
            )
        except Exception as e:
            error_message = f"Agent 루프 실행 중 오류 발생: {e}"
            print(f"[❌ Agent 실행 오류] {error_message}")
            response_logs.append(f"ERROR: {error_message}")
            final_answer = f"요청 처리 중 오류가 발생했습니다: {error_message}. 로그를 확인해주세요."
            break
    if final_answer is None:
        final_answer = "최대 반복 횟수에 도달했지만 최종 답변을 얻지 못했습니다. 요청을 재정의해 주세요."
    return {
        "status": "success" if "오류" not in final_answer else "failed",
        "response": final_answer,
        "logs": response_logs
    }
    
@router.post("/gemini")
async def gemini(request: Request):
    # ... (내용 동일) ...
    try:
        data = await request.json()
        user_input = data.get("prompt", "")
        mode = await asyncio.to_thread(classify_prompt_type, user_input)
        print(f"[🔍 모드 분류 결과] 사용자 입력: '{user_input}' → 모드: {mode}")
        if mode == "system":
            execution_result = await _process_agent_request(user_input)
            if execution_result["status"] == "success":
                return JSONResponse({"response": execution_result["response"], "logs": execution_result["logs"]})
            else:
                return JSONResponse({"response": execution_result["response"], "logs": execution_result["logs"]}, status_code=500)
        else:
            full_prompt = build_prompt(user_input, mode="default")
            response = await call_gemini_agent([types.Content(parts=[types.Part(text=full_prompt)])], use_tools=False, temperature=0.7)
            return JSONResponse({"response": response.text})
    except Exception as e:
        print(f"[❌ 전역 에러] {e}")
        return JSONResponse({"error": str(e)}, status_code=500)