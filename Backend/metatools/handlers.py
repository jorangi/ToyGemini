from __future__ import annotations
import asyncio
import datetime
import json5, chromadb
from sqlalchemy import text
from sql_router import SessionLocal
from llm_utils import call_gemini_agent
from google.genai import types # type: ignore
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json, re
from config import handle_reload_skills
from optimization_manager import OptimizationManager
from typing import Dict, Any, Tuple
from preflight_enforcer import enforce_internal_calls, preflight_check
from config import tool_manager, schema_to_dict


optim_manager = OptimizationManager()
ROOT_DIR = Path("E:/Develop/ToyGemini")
BACKEND_DIR = ROOT_DIR / "Backend" 
TOOL_DEFINITIONS_PATH = BACKEND_DIR / "tools/definitions.json"
GENERATED_SKILLS_PATH = BACKEND_DIR / "tools/generated_skills.py"
GENERATED_DEFINITIONS_PATH = BACKEND_DIR / "tools/generated_definitions.json"
VECTOR_DB_PATH = str(BACKEND_DIR / 'vector_db')
EMBEDDING_MODEL_NAME = 'jhgan/ko-sroberta-multitask'
OPTIMIZATION_THRESHOLD = 3 
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
vector_db_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
tool_collection = vector_db_client.get_or_create_collection(
    name="kaede_tools",
    metadata={"hnsw:space": "cosine"}
)
TOOL_DEFINITION_EXAMPLE = """
[입력 워크플로우 예시]
1. get_user_profile (사용자 정보 조회)
2. send_email (이메일 발송)

[출력 명세 예시]
{
  "name": "send_email_to_user_by_profile",
  "description": "사용자의 프로필 정보를 기반으로 이메일을 보냅니다. 사용자 이름만으로 이메일 주소를 자동으로 찾아 전송할 수 있습니다.",
  "parameters": {
    "type": "object",
    "properties": {
      "user_name": {
        "type": "string",
        "description": "정보를 조회할 사용자의 이름. 예: '신종혁'"
      },
      "subject": {
        "type": "string",
        "description": "이메일의 제목."
      },
      "body": {
        "type": "string",
        "description": "이메일의 본문 내용."
      }
    },
    "required": ["user_name", "subject", "body"]
  }
}
"""

async def generate_tool_definition_and_code(workflow_pattern: str, user_goal: str):
    """
    [✨ 최종 수정] LLM을 호출하여 새 도구의 명세(definition)와 코드(code)를 모두 생성합니다.
    - 스키마(generated_definitions.json)와 코드 시그니처가 정확히 일치하도록 '계약 주도' 규칙을 강제
    - preflight_enforcer.preflight_check 로 등록 전 검사(시그니처/내부호출/드라이런) 수행
    - 실패 시 사유를 LLM 프롬프트에 다시 주입해 자동 재생성 루프
    """
    import asyncio, json, json5
    from pathlib import Path
    from preflight_enforcer import preflight_check
    from config import tool_manager, load_model_priority
    from llm_utils import call_gemini_agent
    from google.genai import types

    from optimization_manager import OptimizationManager
    _om = OptimizationManager()
    if _om.get_count(workflow_pattern) < _om.threshold:
        raise Exception(f"threshold_not_met: count={_om.get_count(workflow_pattern)}, threshold={_om.threshold}")
    print(f"[Optimization Log] 새 도구 명세 생성을 시작합니다... (패턴: {workflow_pattern})")
    optimization_models = load_model_priority()

    # 현재 사용 가능한 도구 목록(정의)을 모델에 제공
    def schema_to_dict(pydantic_schema):
        try:
            return json.loads(pydantic_schema.json())
        except Exception:
            try:
                return pydantic_schema
            except Exception:
                return {}

    available_tools_for_prompt = []
    for d in getattr(tool_manager, "all_definitions", []):
        name = getattr(d, "name", None) if not isinstance(d, dict) else d.get("name")
        description = getattr(d, "description", None) if not isinstance(d, dict) else d.get("description")
        params = getattr(d, "parameters", None) if not isinstance(d, dict) else d.get("parameters")
        available_tools_for_prompt.append({
            "name": name,
            "description": description,
            "parameters": schema_to_dict(params) if params is not None else {}
        })

    CONTRACT_HEADER = """
[계약(필수)]
- 함수 파라미터 이름은 생성된 도구 명세의 parameters.properties 키와 **완전히 동일**해야 한다.
- 함수는 async 로 작성한다. `**kwargs`, `*args` **금지**.
- 내부에서 다른 도구를 호출할 때도 해당 도구의 정의된 키만 명시적으로 전달한다. 메타키(thought 등) 금지.
- 함수 시그니처 예: async def {tool_name}(action_handlers, {param_list}): ...
- 반환은 dict로 하며 최소 {"status":"ok"} 구조를 포함한다.
"""

    # 1) 도구 정의(스키마) 생성 프롬프트
    prompt_for_definition = f"""당신은 시스템 아키텍트입니다. 아래 정보를 바탕으로, 반복적인 작업을 처리하는 하나의 새로운 '통합 도구'를 설계해 주세요.

[분석 대상 워크플로우 패턴]: {workflow_pattern}
[사용자의 최초 목표 예시]: "{user_goal}"

[미션]
이 워크플로우를 하나의 새로운 도구로 정의하고, 작업 간의 종속성을 분석하여 아래 JSON 형식에 맞춰 **JSON 객체만을 반환**해 주세요.

**[가장 중요한 미션: 재사용성을 높이는 '설명(description)' 작성법]**
- **기술적인 동작 방식(How)이 아닌, 이 도구로 '무엇을 할 수 있는지(What)'를 설명해야 합니다.**
- '사용자의 최초 목표 예시'를 참고하여, **"팬사이트 만들기", "커뮤니티 게시글 생성", "HTML 페이지 제작" 등 이 도구가 해결할 수 있는 다양한 사용자 목표들을 포괄하는 일반적인 설명**을 작성하세요.
- "파일을 백업하고 쓴다" 와 같은 기술적인 설명은 절대 사용하지 마세요.

[작명 규칙]
- new_tool_name의 값은 반드시 영문(English)과 밑줄(_)만을 사용한 snake_case 형식으로 작성해야 합니다. (예: "create_web_content_safely")

[출력 형식 (반드시 이 구조를 따르세요)]
{{
  "new_tool_name": "...",
  "description": "...",
  "tags": ["...", "..."],
  "parameters": {{
    "type": "object",
    "properties": {{
      "param1": {{ "type": "string", "description": "..." }}
    }},
    "required": ["param1"]
  }},
  "dependency_graph": {{
    "nodes": [
      {{
        "id": "unique_node_id_1",
        "tool_name": "실행할_기존_도구_이름_1",
        "parameter_mapping": {{ "기존_도구의_파라미터명": "parameters.param1" }}
      }}
    ],
    "edges": []
  }}
}}

[참고] 현재 시스템이 가진 도구 정의 일부:
{json.dumps(available_tools_for_prompt, indent=2, ensure_ascii=False)}
"""

    tool_definition = None
    definition_str = ""
    for attempt in range(3):
        try:
            print(f"[Optimization Log] 명세 생성 시도 ({attempt + 1}/3)...")
            response, _ = await call_gemini_agent(
                [types.Content(role="user", parts=[types.Part(text=prompt_for_definition)])],
                available_models=optimization_models,
                use_tools=False
            )
            if response and hasattr(response, "text") and response.text:
                definition_str = response.text.strip()
                if definition_str.startswith("```json"):
                    definition_str = definition_str[7:].strip()
                if definition_str.startswith("```"):
                    definition_str = definition_str[3:].strip()
                if definition_str.endswith("```"):
                    definition_str = definition_str[:-3].strip()

                tool_definition = json5.loads(definition_str)
                if not tool_definition.get("new_tool_name") and tool_definition.get("name"):
                    tool_definition["new_tool_name"] = tool_definition["name"]
                print(f"[Optimization Log] 새 도구 명세 생성 완료: {tool_definition.get('new_tool_name')}")
                break
            else:
                print(f"⚠️ 명세 생성 실패 (시도 {attempt + 1}): LLM이 빈 응답을 반환했습니다.")
                await asyncio.sleep(3)
        except Exception as e:
            print(f"⚠️ 명세 생성 중 오류 발생 (시도 {attempt + 1}): {e}")
            if attempt < 2:
                await asyncio.sleep(3)
            else:
                print(f"[Optimization Error] 새 도구 명세 생성 최종 실패: {e}\nRaw Response: {definition_str}")
                raise

    if not tool_definition:
        raise Exception("모든 시도 후에도 LLM으로부터 유효한 명세를 받지 못했습니다.")

    # 2) 코드 생성 프롬프트(계약 강조)
    print(f"[Optimization Log] Python 코드 생성을 시작합니다...")

    def _extract_allowed_keys(defn):
        try:
            from google.genai import types as genai_types
            if hasattr(defn, "parameters"):
                sch = defn.parameters
                props = getattr(sch, "properties", {}) or {}
                return sorted(list(props.keys()))
        except Exception:
            pass
        params = (defn.get("parameters") or {}) if isinstance(defn, dict) else {}
        props = params.get("properties") or {}
        return sorted(list(props.keys()))

    # 프롬프트용 PARAM INVENTORY는 단일 진실원(defs_index)에서 생성한다.
    from config import tool_manager
    defs_index = getattr(tool_manager, "defs_index", {})  # {"tool_name": {"allowed": set(...), "required": set(...)}}

    _param_inventory = {}
    for tool_name, meta in defs_index.items():
        allowed = sorted(list(meta.get("allowed", set())))
        required = sorted(list(meta.get("required", set())))
        _param_inventory[tool_name] = {
            "required": required,
            # 타입은 프롬프트 가이드용이면 충분하므로 "any"로 통일(실제 검증은 preflight가 수행)
            "properties": {k: "any" for k in allowed}
        }

    param_inventory_json = json.dumps(_param_inventory, ensure_ascii=False, indent=2)

    # 함수 시그니처 안내에 넣을 파라미터 이름 리스트(생성 대상 도구의 schema 기반)
    _param_keys = list(((tool_definition.get("parameters") or {}).get("properties") or {}).keys())
    sig_param_list = ", ".join(_param_keys)
    # === (추가) FunctionDeclaration.parameters → PARAM INVENTORY(JSON) 생성 ===
    import json

    # 종료성 도구 제거
    _bannable = {"final_response", "ask_followup_question", "clarify", "reflect"}
    _all_defs = [d for d in getattr(tool_manager, "all_definitions", []) if getattr(d, "name", None)]
    _defs_no_terminals = [d for d in _all_defs if getattr(d, "name") not in _bannable]

    def _schema_to_dict(schema):
        if not schema:
            return {}
        if isinstance(schema, dict):
            return schema
        # google types.Schema 같은 객체 호환
        try:
            return schema.to_dict() if hasattr(schema, "to_dict") else dict(schema)
        except Exception:
            return {}

    def _schema_to_inventory(schema):
        sd = _schema_to_dict(schema)
        props = sd.get("properties", {}) or {}
        req = sd.get("required", []) or []
        out_props = {}
        for k, v in props.items():
            t = (v or {}).get("type")
            if isinstance(t, list):  # ["string","null"] 같은 케이스
                t = t[0]
            out_props[k] = str(t or "any")
        return {"required": list(req), "properties": out_props}

    _param_inventory = {
        getattr(fd, "name"): _schema_to_inventory(getattr(fd, "parameters", {}))
        for fd in _defs_no_terminals
        if getattr(fd, "name", None)
    }
    param_inventory_json = json.dumps(_param_inventory, ensure_ascii=False, indent=2)
    # ↓↓↓ 여기 추가
    # (A) base_code_prompt가 기대하는 이름으로 alias
    PARAM_INVENTORY_JSON = param_inventory_json

    # (B) 프롬프트용 allowed_keys_dump 생성
    allowed_keys_dump = "\n".join(
        f"- {name}: allowed={sorted(list(inv['properties'].keys()))}, "
        f"required={sorted(list(inv['required']))}"
        for name, inv in _param_inventory.items()
    )

    # (C) Preflight에서 쓸 defs_index (set 타입)
    defs_index = {
        name: {
            "allowed": set(inv["properties"].keys()),
            "required": set(inv["required"]),
        }
        for name, inv in _param_inventory.items()
    }
    # === (추가 끝) ===
    base_code_prompt = f"""{CONTRACT_HEADER}

    당신은 비동기(asyncio) 파이썬 코드 전문가입니다. 아래의 도구 정의를 바탕으로, 실제 실행될 단일 함수 코드를 작성해 주세요.

    [생성할 도구의 명세]:
    {json5.dumps(tool_definition, indent=2, ensure_ascii=False)}

    [사용 가능한 모든 도구 목록 (기본 + 학습된 도구) — 허용/필수 키 스냅샷]
    {allowed_keys_dump}

    [STRICT PARAMS — 반드시 준수]
    - 아래 인벤토리에 기재된 각 도구의 allowed/required 키만 사용.
    - 내부 호출은 action_handlers["TOOL"](키=값) 형태로만 호출.
    - thought/notes/debug 등 메타키 전달 금지.
    [PARAM INVENTORY(JSON)]
    {PARAM_INVENTORY_JSON}

    [강제 규칙]
    1) 함수 시그니처는 parameters.properties의 키와 **정확히 동일**해야 합니다.
    예: async def {tool_definition.get('new_tool_name')}(action_handlers, {{쉼표로 나열된 파라미터}}):
    2) 내부 도구 호출은 action_handlers['도구명'](**args)로만 하며, 인벤토리에 **없는 키는 절대 사용 금지**.
    3) 코드 외 텍스트 금지. 오직 파이썬 코드만.

    [출력 형식]
    ```python
    # 파이썬 코드만
    """
    base_code_prompt = base_code_prompt.replace("{{PARAM_INVENTORY_JSON}}", param_inventory_json)

    def _resolve_defs_path() -> Path:
        try:
            return GENERATED_DEFINITIONS_PATH  # type: ignore[name-defined]
        except Exception:
            return Path(__file__).parent / "tools" / "generated_definitions.json"

    generated_code = ""
    for attempt in range(3):
        try:
            print(f"[Optimization Log] 코드 생성 시도 ({attempt + 1}/3)...")
            response, _ = await call_gemini_agent(
                [types.Content(role="user", parts=[types.Part(text=base_code_prompt)])],
                available_models=optimization_models,
                use_tools=False
            )
            if response and hasattr(response, "text") and response.text:
                generated_code = response.text.strip()
                if generated_code.startswith("```python"):
                    generated_code = generated_code[9:].strip()
                if generated_code.startswith("```"):
                    generated_code = generated_code[3:].strip()
                if generated_code.endswith("```"):
                    generated_code = generated_code[:-3].strip()
                print(f"[Optimization Log] Python 코드 생성 완료.")
                break
            else:
                print(f"⚠️ 코드 생성 실패 (시도 {attempt + 1}): LLM이 빈 응답을 반환했습니다.")
                await asyncio.sleep(3)
        except Exception as e:
            print(f"⚠️ 코드 생성 중 오류 발생 (시도 {attempt + 1}): {e}")
            if attempt < 2:
                await asyncio.sleep(3)
            else:
                print(f"[Optimization Error] Python 코드 생성 최종 실패: {e}\nRaw Response: {generated_code}")
                raise

    if not generated_code:
        raise Exception("모든 시도 후에도 LLM으로부터 유효한 코드를 받지 못했습니다.")

    # 3) Preflight: 실패 시 사유를 넣어 재생성. 통과 시 종료
    defs_path = _resolve_defs_path()
    MAX_PREFLIGHT_RETRY = 3
    ok = False
    reason = ""
    for attempt in range(1, MAX_PREFLIGHT_RETRY + 1):
        print(f"[Preflight Debug] checking new tool '{tool_definition.get('new_tool_name')}'")
        ok, reason = await preflight_check(
            tool_definition, generated_code, defs_path,
            defs_index=getattr(tool_manager, "defs_index", {})
        )
        if ok:
            print("[Optimization Log] Preflight 통과")
            break
        print(f"[Optimization Log] Preflight 실패({attempt}/{MAX_PREFLIGHT_RETRY}): {reason}")

        # --- Preflight 자동 수선 블록 (파라미터 키 교정 전용) ---
        if not re.search(r"(알 수 없는 키|unknown key)", reason, re.IGNORECASE):
            # 파라미터 키 이슈가 아니면 전체 재생성
            fix_prompt = base_code_prompt + f"\n\n# Preflight 사유:\n# {reason}\n"
            resp, _ = await call_gemini_agent([types.Content(role="user", parts=[types.Part(text=fix_prompt)])])
            new_code = getattr(resp, "text", "") or ""
            if new_code.strip():
                generated_code = new_code
            continue
        # 1) reason에서 '문제 도구'와 '잘못된 키'를 추출(가능할 때만)
        bad_tool = None
        bad_keys = []
        try:
            mt = re.search(r"내부 호출\s+([A-Za-z0-9_]+)", reason)
            if mt:
                bad_tool = mt.group(1)
            mk = re.search(r"\[([^\]]+)\]", reason)
            if mk:
                bad_keys = [k.strip(" '\"\t\n\r") for k in mk.group(1).split(",")]
        except Exception:
            pass

        # 2) 현재 레지스트리에서 허용/필수 키 인벤토리 확보
        try:
            _inv = param_inventory_json  # (앞선 단계에서 이미 만든 경우)
        except NameError:
            # 없으면 즉석 빌드
            from config import tool_manager
            def _schema_to_dict(s):
                try:
                    t = getattr(s, "type", None)
                    props = getattr(s, "properties", None)
                    req = getattr(s, "required", None)
                    out = {"type": t if isinstance(t, str) else "object", "properties": {}, "required": []}
                    if isinstance(props, dict):
                        out["properties"] = {k: {"type": getattr(v, "type", "string")} for k, v in props.items()}
                    if isinstance(req, (list, tuple)):
                        out["required"] = list(req)
                    return out
                except Exception:
                    return {"type":"object","properties":{},"required":[]}
            _inv = {}
            for d in getattr(tool_manager, "all_definitions", []):
                _name = getattr(d, "name", None)
                _schema = _schema_to_dict(getattr(d, "parameters", None))
                if _name:
                    _inv[_name] = {
                        "allowed": sorted(list((_schema.get("properties") or {}).keys())),
                        "required": sorted(list(_schema.get("required") or [])),
                    }

        allowed = (_inv.get(bad_tool) or {}).get("allowed", [])
        required = (_inv.get(bad_tool) or {}).get("required", [])

        # 3) 수선 프롬프트: 내부 호출부만 정확한 파라미터 키로 수정하게 유도 (전체 재생성 금지)
        if bad_tool and allowed:
            fix_prompt = f"""{CONTRACT_HEADER}

[수정 지침 — 반드시 반영]
- 아래 Preflight 실패 사유를 해결하는 방향으로 코드를 다시 작성하세요.
- 동일 도구명/시그니처 규칙/금지사항은 그대로 유지해야 합니다.

[Preflight 실패 사유]
{reason}

[STRICT PARAMS — 반드시 준수]
- 아래 인벤토리에 기재된 각 도구의 allowed/required 키만 사용.
- parameters.properties 키와 **정확히 일치**하는 인자 목록만 사용
- 내부 도구 호출 시 정의 밖 키 전달 금지
- **kwargs, *args 금지
- 코드 외 설명/코드펜스 금지

[PARAM INVENTORY(JSON)]
{param_inventory_json}

[생성할 도구의 명세]
{json5.dumps(tool_definition, indent=2, ensure_ascii=False)}

[출력 형식]
# (코드만)
"""
        else:
            # 문제 도구를 특정하지 못하면 전체 재생성(단, 기존 규칙/인벤토리를 포함한 base_code_prompt 사용)
            fix_prompt = base_code_prompt + f"\n\n# Preflight 사유:\n# {reason}\n"

        # 4) LLM에 수선 요청 → 생성 코드 교체 후 루프 계속
        resp, _ = await call_gemini_agent([types.Content(role="user", parts=[types.Part(text=fix_prompt)])])
        new_code = getattr(resp, "text", "") or ""
        if new_code.strip():
            generated_code = new_code
        # --- 자동 수선 블록 끝 ---

        try:
            response, _ = await call_gemini_agent(
                [types.Content(role="user", parts=[types.Part(text=fix_prompt)])],
                available_models=optimization_models,
                use_tools=False
            )
            new_code = ""
            if response and hasattr(response, "text") and response.text:
                new_code = response.text.strip()
                if new_code.startswith("```python"):
                    new_code = new_code[9:].strip()
                if new_code.startswith("```"):
                    new_code = new_code[3:].strip()
                if new_code.endswith("```"):
                    new_code = new_code[:-3].strip()

            if new_code:
                generated_code = new_code
            else:
                print("[Optimization Log] 정정 코드가 비어있음. 동일 코드 재시도.")
        except Exception as e:
            print(f"[Optimization Log] 정정 코드 생성 중 오류: {e}")
            # 다음 루프로 진행

    if not ok:
        raise Exception(f"Preflight 재시도 실패: {reason}")

    return tool_definition, generated_code

def _ensure_defs_file_structure(path) -> Dict[str, Any]:
    if not path.exists():
        return {"tools": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "tools" not in data or not isinstance(data["tools"], list):
            return {"tools": []}
        return data
    except Exception:
        return {"tools": []}
def _upsert_tool_definition(defs_json: Dict[str, Any], tool_entry: Dict[str, Any]) -> Dict[str, Any]:
    name = tool_entry.get("name")
    updated = False
    for i, t in enumerate(defs_json.get("tools", [])):
        if t.get("name") == name:
            defs_json["tools"][i] = tool_entry
            updated = True
            break
    if not updated:
        defs_json.setdefault("tools", []).append(tool_entry)
    return defs_json
def _replace_or_append_function(code_path, func_name: str, code_str: str):
    """
    generated_skills.py 안에 같은 함수명이 있으면 교체, 없으면 append.
    """
    if not code_path.exists():
        code_path.write_text("# Auto-generated skills\n\n", encoding="utf-8")

    src = code_path.read_text(encoding="utf-8")

    # 함수 블록 교체: "async def func_name("부터 다음 "async def " 또는 EOF까지
    pattern = rf"(?s)async\s+def\s+{re.escape(func_name)}\s*\(.*?(?=(?:\nasync\s+def\s+)|\Z)"
    if re.search(pattern, src):
        new_src = re.sub(pattern, code_str.strip() + "\n\n", src)
    else:
        new_src = src.rstrip() + "\n\n" + code_str.strip() + "\n\n"

    code_path.write_text(new_src, encoding="utf-8")
async def register_newly_generated_tool(definition: Dict[str, Any], code_str: str) -> Dict[str, Any]:
    """
    1) preflight_check 로 스키마-시그니처 일치 + 내부 호출 파라미터 검증 + 드라이런
    2) 통과 시 generated_definitions.json / generated_skills.py 반영
    3) 실패 시 status=error 로 사유 반환
    """
    # 0) 경로 상수 확인
    try:
        _ = GENERATED_DEFINITIONS_PATH
        _ = GENERATED_SKILLS_PATH
    except NameError as e:
        return {"status": "error", "reason": f"path_constants_missing: {e}"}
    
    _bannable = {"final_response", "ask_followup_question", "clarify", "reflect"}
    _all_defs = [d for d in getattr(tool_manager, "all_definitions", []) if getattr(d, "name", None)]
    _defs_no_terminals = [d for d in _all_defs if getattr(d, "name") not in _bannable]
    def _schema_to_dict(schema):
        if not schema:
            return {}
        if isinstance(schema, dict):
            return schema
        # google types.Schema 같은 객체 호환
        try:
            return schema.to_dict() if hasattr(schema, "to_dict") else dict(schema)
        except Exception:
            return {}

    def _schema_to_inventory(schema):
        sd = _schema_to_dict(schema)
        props = sd.get("properties", {}) or {}
        req = sd.get("required", []) or []
        out_props = {}
        for k, v in props.items():
            t = (v or {}).get("type")
            if isinstance(t, list):  # ["string","null"] 같은 케이스
                t = t[0]
            out_props[k] = str(t or "any")
        return {"required": list(req), "properties": out_props}

    _param_inventory = {
    getattr(fd, "name"): _schema_to_inventory(getattr(fd, "parameters", {}))
    for fd in _defs_no_terminals
    if getattr(fd, "name", None)
    }
    param_inventory_json = json.dumps(_param_inventory, ensure_ascii=False, indent=2)

    # ✨ 추가: 프롬프트/Preflight에서 동일하게 쓸 인덱스와 덤프
    defs_index = {
        name: {
            "allowed": set(inv.get("properties", {}).keys()),
            "required": set(inv.get("required", [])),
        }
        for name, inv in _param_inventory.items()
    }

    allowed_keys_dump = "\n".join(
        f"- {name}: allowed={sorted(list(spec['allowed']))}, required={sorted(list(spec['required']))}"
        for name, spec in sorted(defs_index.items())
    )

    PARAM_INVENTORY_JSON = json.dumps(
        {name: {"required": sorted(list(v["required"])),
                "properties": sorted(list(v["allowed"]))}
        for name, v in defs_index.items()},
        ensure_ascii=False, indent=2
    )

    # (C) Preflight에서 쓸 defs_index (set 타입)
    defs_index = {
        name: {
            "allowed": set(inv["properties"].keys()),
            "required": set(inv["required"]),
        }
        for name, inv in _param_inventory.items()
    }
    # 1) Preflight
    ok, reason = await preflight_check(
        definition, code_str, GENERATED_DEFINITIONS_PATH,
        defs_index=defs_index  # ← 동일하게 로컬 인덱스 사용
    )
    if not ok:
        return {"status": "error", "reason": f"preflight_failed: {reason}"}

    # 2) 정의/코드 반영
    new_tool_name = definition.get("new_tool_name") or definition.get("name")
    if not new_tool_name:
        return {"status": "error", "reason": "definition has no new_tool_name/name"}

    # definitions.json 업데이트 (name 필드로 저장)
    defs_json = _ensure_defs_file_structure(GENERATED_DEFINITIONS_PATH)
    tool_entry = {
        "name": new_tool_name,
        "description": definition.get("description", ""),
        "parameters": definition.get("parameters", {"type": "object", "properties": {}, "required": []}),
    }
    # 선택: tags나 기타 부가 필드 있으면 보존
    if "tags" in definition:
        tool_entry["tags"] = definition["tags"]

    defs_json = _upsert_tool_definition(defs_json, tool_entry)
    GENERATED_DEFINITIONS_PATH.write_text(json.dumps(defs_json, ensure_ascii=False, indent=2), encoding="utf-8")

    # skills.py 함수 등록/교체
    _replace_or_append_function(GENERATED_SKILLS_PATH, new_tool_name, code_str)
    return {"status": "ok", "tool_name": new_tool_name}
def remove_function_by_name(src: str, func_name: str) -> str:
    """
    매우 단순한 제거기:
    - 'async def {func_name}(' 로 시작하는 라인부터
    다음 'async def ' 시작 라인(또는 파일 끝)까지 제거.
    - 정교한 AST 파싱은 아니지만 충돌을 줄이며, 동일 함수 중복 정의를 방지.
    """
    if not src:
        return ""

    lines = src.splitlines()
    out_lines = []
    removing = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not removing:
            if stripped.startswith(f"async def {func_name}("):
                # 이 라인부터 제거 시작
                removing = True
                continue
            else:
                out_lines.append(line)
        else:
            # 제거 중: 다음 함수 시작 시 제거 종료
            if stripped.startswith("async def "):
                removing = False
                out_lines.append(line)
            # 아니면 계속 스킵

    # 파일 끝까지 제거되었을 수도 있으니 그대로 합침
    return "\n".join(out_lines)
async def run_optimization_workflow(action_input: dict):
    """
    기존과 동일한 시그니처를 유지하되,
    내부 처리를 OptimizationManager에 위임한다.
    """ 
    try:
        workflow_str_list = action_input.get("completed_workflow", [])
        user_goal = action_input.get("user_goal")
        result = await optim_manager.record_and_maybe_materialize(
            workflow_str_list,
            user_goal,
            generate_tool_definition_and_code=generate_tool_definition_and_code,
            register_newly_generated_tool=register_newly_generated_tool,
            handle_reload_skills=handle_reload_skills
        )
        print(f"[Optimization] {result}")
        return result
    except Exception as e:
        return {"status": "error", "reason": f"optimization failed: {e}"}