# prompt_builder.py

from pathlib import Path
from google import genai
import re
import os
import dotenv
import json5

dotenv.load_dotenv()

def extract_json_from_text(text: str) -> str:
    """
    Gemini 응답에서 JSON 객체만 추출합니다.
    마크다운 코드 블록(```json ... ```) 또는 텍스트 내 JSON 전체를 견고하게 처리합니다.
    json5 모듈을 사용하여 파싱 견고성을 높입니다.
    """
    json_content_candidate = ""

    # 1. 마크다운 코드 블록 내의 JSON을 우선적으로 추출 시도
    # `re.DOTALL`은 개행 문자도 .에 포함하도록 하고, `re.IGNORECASE`는 'json' 대소문자를 무시합니다.
    match_code_block = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if match_code_block:
        json_content_candidate = match_code_block.group(1).strip()
        try:
            # json5.loads를 사용하여 파싱 시도
            json5.loads(json_content_candidate)
            return json_content_candidate # 유효하면 바로 반환
        except ValueError: # json5.loads는 파싱 실패 시 ValueError를 발생시킵니다.
            print(f"[WARN] 마크다운 블록 내 JSON5 파싱 실패 (내용 문제): {json_content_candidate[:100]}...")
            # 파싱 실패 시 다음 로직으로 넘어감
            pass

    # 2. 마크다운 코드 블록이 없거나 유효하지 않다면,
    # 'Thought'와 'Action' 키를 포함하는 최상위 JSON 객체를 찾아 추출
    # 이 패턴은 AI 응답의 전형적인 구조를 따릅니다.
    match_thought_action_json = re.search(
        r"\{\s*\"Thought\":\s*\".*?\",\s*\"Action\":\s*\{.*\}\s*\}", # Thought와 Action을 포함하는 JSON 구조
        text, re.DOTALL
    )
    if match_thought_action_json:
        json_content_candidate = match_thought_action_json.group(0).strip()
        try:
            json5.loads(json_content_candidate)
            return json_content_candidate # 유효하면 바로 반환
        except ValueError:
            print(f"[WARN] Thought/Action JSON5 파싱 실패: {json_content_candidate[:100]}...")
            pass

    # 3. Fallback: 위에 모든 방법으로도 찾지 못했다면, 텍스트 전체에서 가장 바깥 JSON 객체를 탐욕적으로 추출
    match_fallback = re.search(r"(\{.*\})", text, re.DOTALL)
    if match_fallback:
        json_content_candidate = match_fallback.group(1).strip()
        try:
            json5.loads(json_content_candidate)
            return json_content_candidate # 유효하면 바로 반환
        except ValueError:
            print(f"[WARN] Fallback JSON5 파싱 실패: {json_content_candidate[:100]}...")
            pass

    # 4. 어떤 JSON5도 찾지 못하면 빈 문자열 반환
    return ""

def build_prompt(user_input: str) -> str:
    """
    핵심 규칙 파일들을 기반으로 AI의 역할과 사고방식을 정의합니다.
    """
    rules_dir = Path(__file__).parent / "gemini_rules"
    
    core_filenames = [
        "final_rules.txt"
    ]

    forceful_instruction = ""
    all_rules_content = [forceful_instruction]
    
    for filename in core_filenames:
        rule_path = rules_dir / filename
        try:
            with open(rule_path, "r", encoding="utf-8") as f:
                all_rules_content.append(f.read().strip())
        except FileNotFoundError:
            # 파일이 없어도 오류 없이 계속 진행합니다.
            print(f"경고: 규칙 파일 '{filename}'을 찾을 수 없습니다.")

    combined_rules = "\n\n---\n\n".join(all_rules_content)

    return f"{combined_rules}\n\n---\n\n[사용자 요청]\n{user_input}"
def build_plan_prompt(goal: str) -> str:
    return f"""
        사용자의 목표를 달성하기 위해 필요한 단계를 순서대로 나열해줘.
        각 단계는 간결하고 명확하게 설명해줘.

        사용자 목표: "{goal}"

        출력 형식은 반드시 아래 JSON처럼 해줘:

        ```json
        {{
        "Thought": "...단계 설명...",
        "Action": {{
            "tool_name": "plan",
            "parameters": {{}}
        }}
        }}
        ```
        """
