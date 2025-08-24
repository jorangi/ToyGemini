import os
import dotenv
from google import genai
from google.genai import types
from google.api_core import exceptions

# 이 파일에서 직접 필요한 설정만 가져옵니다.
dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")

# 비동기 클라이언트를 여기서 생성
client = genai.Client()

async def call_gemini_agent(
    prompt_content: list[types.Content],
    use_tools: bool = True,
    temperature: float = 0.2,
    tools=None,
    available_models: list[str] = None
) -> tuple[types.GenerateContentResponse, str | None]:
    
    if available_models is None or not available_models:
        # 만약 available_models가 없으면 기본 우선순위 리스트 사용 (초기 호출 또는 fallback)
        available_models = [m.strip() for m in os.getenv("GEMINI_MODEL_PRIORITY_LIST").split(',')]
    
    #print("📤 Gemini 요청 프롬프트:", prompt_content)
    # (추가) 메시지 파트 정규화: 파일/바이너리는 텍스트로 강등
    def _as_content(obj):
        # 이미 Content 타입이면 그대로
        if hasattr(obj, "parts") and hasattr(obj, "role"):
            return obj
        # 문자열은 텍스트 파트로 감쌈
        if isinstance(obj, str):
            return types.Content(role="user", parts=[types.Part(text=obj)])
        # text 속성이 있으면 그걸로
        text = getattr(obj, "text", None)
        if isinstance(text, str):
            return types.Content(role="user", parts=[types.Part(text=text)])
        # dict/list 등은 문자열화
        return types.Content(role="user", parts=[types.Part(text=str(obj))])

    def _sanitize_messages(msgs):
        safe = []
        for m in msgs or []:
            c = _as_content(m)
            parts = []
            for p in (getattr(c, "parts", None) or []):
                # 파일/바이너리/URI 파트는 텍스트로 강등
                if getattr(p, "inline_data", None) is not None or getattr(p, "file_data", None) is not None:
                    parts.append(types.Part(text="[binary omitted]"))
                # 함수 호출 파트는 유지(도구 호출 컨텍스트 보존)
                elif getattr(p, "function_call", None) is not None:
                    parts.append(types.Part(function_call=p.function_call))
                # 평범한 텍스트
                elif getattr(p, "text", None) is not None:
                    parts.append(types.Part(text=p.text))
                else:
                    parts.append(types.Part(text=str(p)))
            if not parts:
                parts = [types.Part(text="")]
            safe.append(types.Content(role=getattr(c, "role", "user"), parts=parts))
        return safe

    # 여기서 한 번 정규화
    safe_prompt = _sanitize_messages(prompt_content)
    last_exception = None
    for model_name in available_models:
        print(f"[🔄 모델 시도] '{model_name}' 모델로 API 호출을 시도합니다.")
        
        try:
            from os import getenv
            max_out = int(getenv("GEMINI_MAX_OUTPUT_TOKENS", "32768"))  # 필요시 65536로
            config_args = {
                "max_output_tokens": max_out,
                "temperature": temperature,
                "candidate_count": 1,               # 불필요한 후보 생성 금지
                "response_mime_type": "text/plain", # 텍스트 전용으로 길게
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
                contents=safe_prompt,
                config=types.GenerateContentConfig(**config_args)
            )
            
            #print("📥 Gemini 응답 전체:", response)
            print(f"[✅ 모델 성공] '{model_name}' 모델 호출에 성공했습니다.")
            return response, model_name # ✨ 성공한 모델 이름 반환

        except (exceptions.ResourceExhausted, exceptions.TooManyRequests, exceptions.ServiceUnavailable) as e:
            print(f"[⚠️ 모델 실패] '{model_name}' 모델의 할당량이 소진되었습니다. 다음 모델을 시도합니다.")
            last_exception = e
            continue
        except Exception as e:
            print(f"[❌ API 오류] '{model_name}' 모델에서 예상치 못한 오류가 발생했습니다: {type(e)}, {e}")
            last_exception = e
            continue
            
    print(f"[❌ 모든 모델 실패] 사용 가능한 모든 모델의 호출에 실패했습니다. 최종 오류: {last_exception}")
    error_message = f"모든 API 모델 호출에 실패하여 요청을 처리할 수 없습니다. 잠시 후 다시 시도해주세요. (최종 오류: {str(last_exception)})"
    
    error_action = types.FunctionCall(
        name='final_response',
        args={'answer': error_message, 'thought': "모든 우선순위의 모델에서 API 호출이 실패하여 작업을 중단합니다."}
    )
    
    error_response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(parts=[types.Part(function_call=error_action)])
            )
        ]
    )
    return error_response, None
