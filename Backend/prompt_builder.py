from pathlib import Path
from google import genai
import re
import os
import dotenv
from google.api_core import exceptions # [추가] 429 에러 처리를 위한 임포트

# [추가] .env 파일 로드 및 모델 목록 설정
dotenv.load_dotenv()
MODEL_PRIORITY_LIST = os.getenv("GEMINI_MODEL_PRIORITY_LIST", "gemini-1.5-flash-latest").split(',')


def extract_json_from_text(text: str) -> str:
    """
    Gemini 응답에서 JSON 객체만 추출합니다.
    마크다운 코드 블록(```json ... ```)을 포함한 다양한 형식을 처리합니다.
    """
    # 1. 마크다운 코드 블록에서 JSON 추출 시도 (```json ... ``` 또는 ``` ... ```)
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)

    # 2. 코드 블록이 없다면, 텍스트에서 직접 JSON 객체 검색
    match = re.search(r"(\{.*?\})", text, re.DOTALL)
    if match:
        return match.group(1)

    # 3. 어떤 JSON도 찾지 못하면 빈 문자열 반환
    return ""

# 기존 Prompt Builder
def build_prompt(user_input: str, mode: str = "default") -> str:
    rule_file = {
        "default": "default_rules.txt",
        "system": "system_rules.txt"
    }.get(mode, "default_rules.txt")

    rules_dir = Path(__file__).parent / "gemini_rules"
    rule_path = rules_dir / rule_file

    try:
        with open(rule_path, "r", encoding="utf-8") as f:
            rules = f.read().strip()
    except FileNotFoundError:
        rules = ""

    return f"{rules}\n\n사용자 요청:\n{user_input}"


# [✨ 핵심 수정] 모델 폴백 기능이 적용된 자동 규칙 분류기
def classify_prompt_type(user_input: str) -> str:
    """
    사용자의 요청 의도를 'system' 또는 'default'로 분류한다.
    주력 모델의 할당량 소진 시 예비 모델을 사용하여 자동 전환한다.
    """
    classification_prompt = f"""
당신은 사용자의 요청 의도를 'system' 또는 'default'로 분류하는 전문가입니다.

'system'은 데이터베이스, 파일 시스템 조작, 또는 게임 상태(State)를 변경해야 하는 모든 요청을 의미합니다.
'default'는 시스템 상태와 무관한 일반적인 대화나 질문을 의미합니다.

아래 예시를 보고 사용자의 요청을 가장 적절하게 분류하십시오.

[분류 예시]
- "DB에서 홍길동 조회해줘" → system (직접적인 DB 명령)
- "test.txt 파일 내용 보여줘" → system (직접적인 파일 명령)
- "오늘 날씨 어때?" → default (상태와 무관한 질문)
- "너는 누구야?" → default (페르소나에 대한 질문)
- "TRPG 게임을 시작하고 싶어." → system (게임 상태 초기화 및 DB 준비가 필요함)
- "내 캐릭터를 만들래." → system (캐릭터 정보를 DB에 생성 및 저장해야 함)
- "상인에게 말을 걸어보자." → system (NPC 상호작용은 게임 상태를 변경하고 DB를 조회/수정할 수 있음)
- "안녕? 반가워." → default (단순 인사)

[분류할 사용자 요청]
\"\"\"{user_input}\"\"\"

[출력]
오직 'system' 또는 'default' 중 하나만 출력하십시오.
""".strip()

    client = genai.Client()
    
    # 모델 목록을 순회하며 API 호출 시도
    for model_name in MODEL_PRIORITY_LIST:
        model_name = model_name.strip()
        try:
            print(f"[🧠 classify_prompt_type] '{model_name}' 모델로 분류 시도...")
            result = client.models.generate_content(model=model_name, contents=classification_prompt)
            text = result.text.strip().lower()
            
            print(f"[🧠 classify_prompt_type] '{user_input}' → 판단 결과: {text}")

            if "system" in text:
                return "system"
            return "default"
        
        except exceptions.ResourceExhausted as e:
            print(f"[⚠️ 분류기 모델 실패] '{model_name}' 모델 할당량 소진. 다음 모델로 전환합니다.")
            continue # 다음 모델로 계속
            
        except Exception as e:
            print(f"[❌ 분류기 오류] '{model_name}' 모델에서 예상치 못한 오류 발생: {e}")
            # 다른 종류의 에러가 발생하면 더 이상 시도하지 않고 기본값으로 처리
            break

    # 모든 모델 호출에 실패한 경우, 안전하게 'default'로 처리
    print("[⚠️ 분류기 최종 실패] 모든 모델 호출에 실패하여 'default' 모드로 처리합니다.")
    return "default"


# 최종 요청 빌더 (수정 없음)
def build_auto_prompt(user_input: str) -> str:
    mode = classify_prompt_type(user_input)
    return build_prompt(user_input, mode)