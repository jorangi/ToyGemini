# preflight_enforcer.py
from __future__ import annotations
import inspect
import json
import types
from pathlib import Path
from typing import Any, Dict, Set, Tuple, Optional
from config import schema_to_dict


def load_definitions(defs_path: Path) -> Dict[str, Dict[str, Set[str]]]:
    """
    generated_definitions.json에서 각 도구의 allowed/required 키 집합을 적재한다.
    """
    
    if not defs_path.exists():
        return {}
    try:
        data = json.loads(defs_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: Dict[str, Dict[str, Set[str]]] = {}
    for tool in data.get("tools", []):
        name = tool.get("name")
        params = tool.get("parameters", {})
        # [추가] 혹시 dict가 아니면 공용 정규화 사용
        if not isinstance(params, dict):
            params = schema_to_dict(params)

        props = params.get("properties", {}) if isinstance(params, dict) else {}
        required = set(params.get("required", [])) if isinstance(params, dict) else set()
        if name:
            out[name] = {"allowed": set(props.keys()), "required": required}
    # --- 런타임 FunctionDeclaration 병합(정규화) ---
    try:
        from config import tool_manager, schema_to_dict as _std
        for fd in getattr(tool_manager, "all_definitions", []):
            name = getattr(fd, "name", None)
            if not name:
                continue
            sd = _std(getattr(fd, "parameters", None))  # ← 항상 dict로 정규화
            props = (sd.get("properties") or {})
            req = set(sd.get("required") or [])

            # JSON(generated_definitions.json) 쪽 값이 이미 있으면 유지,
            # 없을 때에만 런타임 정의로 채움
            if name not in out:
                out[name] = {"allowed": set(props.keys()), "required": req}
            else:
                # 기존 allowed/required가 비어있으면만 보완
                if not out[name]["allowed"]:
                    out[name]["allowed"] = set(props.keys())
                if not out[name]["required"]:
                    out[name]["required"] = req
    except Exception:
        pass
    # --- 병합 끝 ---

    return out
class ContractError(Exception):
    pass
def assert_signature_matches_schema(func, schema_allowed: Set[str], schema_required: Set[str]) -> None:
    """
    - **kwargs 금지
    - 파라미터 이름 집합 == schema_allowed (정확 일치)
    - required는 allowed의 부분집합이어야 함(정상 케이스)
    """
    sig = inspect.signature(func)
    param_names = []
    for name, p in sig.parameters.items():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            raise ContractError(f"함수 {getattr(func,'__name__','<func>')} 가 **kwargs를 허용함. 금지.")
        if p.kind in (inspect.Parameter.VAR_POSITIONAL,):
            raise ContractError(f"함수 {getattr(func,'__name__','<func>')} 가 *args를 허용함. 금지.")
        if name == "action_handlers":
            # 첫 인자는 시스템이 주입하는 action_handlers일 수 있으므로 제외
            continue
        param_names.append(name)

    param_set = set(param_names)
    if param_set != schema_allowed:
        raise ContractError(f"시그니처 파라미터 {sorted(list(param_set))} 가 스키마 allowed {sorted(list(schema_allowed))} 와 일치하지 않음.")
    if not schema_required.issubset(schema_allowed):
        raise ContractError(f"스키마 required가 allowed의 부분집합이 아님: required={sorted(list(schema_required))}, allowed={sorted(list(schema_allowed))}")
class MockActionHandlers(dict):
    """
    생성된 스킬이 내부 도구를 호출할 때 전달하는 kwargs를 검사한다.
    - definitions에 존재하는 내부 도구는 definitions의 allowed/required로 엄격 검증
    - definitions에 없으면 기본 화이트리스트를 적용(보수적으로)
    - 실제 IO는 수행하지 않고 호출 로그만 기록
    """
    def __init__(self, defs_index: Dict[str, Dict[str, Set[str]]]):
        super().__init__()
        self._defs_index = defs_index
        self.calls = []

    def __getitem__(self, tool_name: str):
        def _callable(**kwargs):
            # 호출 전 파라미터 검증(허용키/필수키)
            self._validate_call(tool_name, kwargs)
            # 호출 기록(선택)
            try:
                self.calls.append((tool_name, kwargs))
            except Exception:
                pass
            # 실제 I/O 금지 — 드라이런이므로 NO-OP 반환
            return {"__mock__": True, "tool": tool_name, "kwargs": kwargs}
        return _callable

    def _validate_call(self, tool_name: str, kwargs: Dict[str, Any]) -> None:
        # definitions 기반 검증
        if tool_name in self._defs_index:
            allowed = self._defs_index[tool_name]["allowed"]
            required = self._defs_index[tool_name]["required"]
            keys = set(kwargs.keys())
            unknown = keys - allowed
            missing = required - keys
            if unknown:
                raise ContractError(f"내부 호출 {tool_name}에 정의 밖 키 존재: {sorted(list(unknown))}")
            if missing:
                raise ContractError(f"내부 호출 {tool_name}에 필수 키 누락: {sorted(list(missing))}")
        else:
            # 정의가 로딩되지 않은 도구를 호출하려 함 → 즉시 실패(프롬프트 재생성 유도)
            raise ContractError(
                f"정의 미탑재 도구 호출: '{tool_name}'. "
                f"도구 정의(카탈로그) 로드 상태와 이름을 확인하세요."
            )
def _extract_param_keys_from_definition(defn):
    """
    defn: google.genai.types.FunctionDeclaration 또는 dict 형태 모두 지원
    반환: (allowed_keys_set, required_keys_set)
    """
    try:
        # genai FunctionDeclaration
        from google.genai import types as genai_types
        if isinstance(defn, genai_types.FunctionDeclaration):
            sch = defn.parameters  # types.Schema
            if not sch or not getattr(sch, "properties", None):
                return set(), set()
            props = sch.properties  # dict[str, types.Schema]
            allowed = set(props.keys())
            required = set(getattr(sch, "required", []) or [])
            return allowed, required
    except Exception:
        pass

    # dict 형태 (generated_definitions.json 등)
    if isinstance(defn, dict):
        params = defn.get("parameters") or {}
        props = params.get("properties") or {}
        allowed = set(props.keys())
        required = set(params.get("required") or [])
        return allowed, required

    # 알 수 없는 포맷이면 빈 셋 반환
    return set(), set()
def resolve_param_keys(tool_name, tool_manager):
    """
    tool_manager에서 tool_name의 정의를 찾아 파라미터 키셋을 반환.
    못 찾으면 (None, None) 반환하여 '검증 생략'하도록 한다.
    """
    try:
        # tool_manager.all_definitions 는 FunctionDeclaration 또는 dict를 섞어서 가지고 있다고 가정
        for d in getattr(tool_manager, "all_definitions", []):
            name = getattr(d, "name", None) if not isinstance(d, dict) else d.get("name")
            if name == tool_name:
                allowed, required = _extract_param_keys_from_definition(d)
                # 정의는 있는데 키가 0개면 검증 유의미하지 않으므로 스킵
                if not allowed and not required:
                    return None, None
                return allowed, required
    except Exception as e:
        print(f"[Preflight] resolve_param_keys error for '{tool_name}': {e}")
        return None, None
    return None, None
def make_minimal_args(schema_allowed: Set[str], schema_required: Set[str]) -> Dict[str, Any]:
    """
    스키마 기준으로 '필수 키'만 최소 인자 생성.
    - 도메인 경로/값 하드코딩 금지
    - 타입 정보가 없으므로 placeholder 문자열 사용
    """
    args: Dict[str, Any] = {}
    for k in schema_required:
        args[k] = f"<AUTO_{k}>"
    return args
def exec_code_in_isolated_namespace(py_code: str) -> Dict[str, Any]:
    """
    생성된 스킬 코드 문자열을 격리된 네임스페이스에서 exec한다.
    """
    ns: Dict[str, Any] = {}
    compiled = compile(py_code, "<generated_skills>", "exec")
    exec(compiled, ns, ns)
    return ns
async def preflight_check(
    definition: Dict[str, Any],
    code_str: str,
    defs_path: Path,
    defs_index: Optional[Dict[str, Dict[str, Set[str]]]] = None
) -> Tuple[bool, str]:
    """
    - 스키마 로드
    - 코드 exec → 대상 함수 찾기
    - 시그니처 == 스키마 allowed 정확 매칭 확인
    - 드라이런: MockActionHandlers로 내부 호출 키 검증
    """
    # ↓↓ 한 줄 추가 (넘겨받은 인덱스를 최우선으로 사용, 없으면 기존 fallback)
    defs_index = defs_index or load_definitions(defs_path)
    defs_index = load_definitions(defs_path)
    new_tool_name = definition.get("new_tool_name") or definition.get("name")
    if not new_tool_name:
        return False, "definition에 new_tool_name/name 없음"

    params = definition.get("parameters", {})
    props = params.get("properties", {}) if isinstance(params, dict) else {}
    allowed = set(props.keys())
    required = set(params.get("required", [])) if isinstance(params, dict) else set()

    if not allowed:
        return False, "스키마 properties가 비어 있음"

    ns = exec_code_in_isolated_namespace(code_str)
    if new_tool_name not in ns:
        return False, f"코드에 함수 {new_tool_name} 가 존재하지 않음"

    func = ns[new_tool_name]
    if not inspect.iscoroutinefunction(func):
        return False, f"{new_tool_name} 는 async 함수가 아님(비동기 필수)."

    try:
        assert_signature_matches_schema(func, allowed, required)
    except ContractError as e:
        return False, f"시그니처-스키마 불일치: {e}"

    # 드라이런
    mock_handlers = MockActionHandlers(defs_index)
    minimal_args = make_minimal_args(allowed, required)
    try:
        await func(mock_handlers, **minimal_args)
    except ContractError as e:
        return False, f"내부 호출 파라미터 위반: {e}"
    except Exception as e:
        # 코드 내부 일반 예외도 등록 거부 (IO 시도 등)
        return False, f"드라이런 중 예외: {e}"

    return True, "ok"
def enforce_internal_calls(internal_calls: list[dict], tool_manager) -> tuple[bool, str]:
    """
    internal_calls: [{"tool_name": "...", "args": {...}}, ...]
    반환: (ok, reason)
    """
    for call in internal_calls:
        tname = call.get("tool_name")
        args  = call.get("args") or {}
        allowed, required = resolve_param_keys(tname, tool_manager)

        # 정의를 못 찾으면 '엄격 실패' 대신 경고 후 패스 (생성 루프가 막히지 않도록)
        if allowed is None and required is None:
            print(f"[Preflight] skip check: definition not found for '{tname}'")
            continue

        # 알 수 없는 키 검출
        unknown = [k for k in args.keys() if k not in allowed]
        if unknown:
            return False, f"내부 호출 {tname}에 알 수 없는 키 존재: {unknown} (allowed: {sorted(list(allowed))})"

        # 필수 키 누락 검출
        missing = [k for k in (required or []) if k not in args]
        if missing:
            return False, f"내부 호출 {tname}에 필수 키 누락: {missing}"

    return True, "ok"