from pathlib import Path
from google import genai
import re
import os
import dotenv

dotenv.load_dotenv()

def extract_json_from_text(text: str) -> str:
    """
    Gemini 응답에서 JSON 객체만 추출합니다.
    마크다운 코드 블록(```json ... ```) 또는 텍스트 내 JSON 전체를 greedy하게 처리합니다.
    """
    # 1. 마크다운 코드 블록 내 JSON을 greedy하게 추출 (가장 바깥 중괄호까지!)
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)

    # 2. 코드 블록이 없다면, 텍스트 전체에서 가장 바깥 JSON을 greedy하게 추출
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        return match.group(1)

    # 3. 어떤 JSON도 찾지 못하면 빈 문자열 반환
    return ""

# [✨ 수정] build_prompt 함수만 남기고 다른 함수들은 삭제합니다.
def build_prompt(user_input: str) -> str:
    """
    항상 system_rules.txt를 기반으로 전체 프롬프트를 구성합니다.
    """
    rules_dir = Path(__file__).parent / "gemini_rules"
    rule_path = rules_dir / "system_rules.txt"

    try:
        with open(rule_path, "r", encoding="utf-8") as f:
            rules = f.read().strip()
    except FileNotFoundError:
        rules = ""

    # gemini_router.py의 _process_agent_request에서 전체 프롬프트 내용을 user_input으로 전달합니다.
    return f"{rules}\n\n{user_input}"