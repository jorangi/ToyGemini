import os
import sys
import json
try:
    from preflight_enforcer import StrictActionHandlers as _StrictActionHandlers
except Exception:
    class _StrictActionHandlers(dict):
        """런타임용 얇은 스텁(Preflight에서 이미 키 검증했으므로 이름 해소용으로만 사용)"""
        pass
import json5
import inspect
import dotenv
import chromadb
import importlib.util
import functools
import asyncio
from pathlib import Path
from google.genai import types
from typing import Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
try:
    from google.protobuf.json_format import MessageToDict as _PB_to_dict
except Exception:
    _PB_to_dict = None


# --- 1. 기본 설정 및 경로 정의 ---
dotenv.load_dotenv()
ROOT_DIR = Path(os.getenv("ROOT_DIR", "E:/Develop/ToyGemini"))
FRONTEND_DIR = ROOT_DIR / "Frontend"
BACKEND_DIR = ROOT_DIR / "Backend"
GENERATED_SKILLS_PATH = BACKEND_DIR / "tools/generated_skills.py"
GENERATED_DEFINITIONS_PATH = BACKEND_DIR / "tools/generated_definitions.json"
VECTOR_DB_PATH = str(BACKEND_DIR / 'vector_db')
EMBEDDING_MODEL_NAME = 'jhgan/ko-sroberta-multitask'
MAX_AGENT_ITERATIONS = 20
AGENT_STATE_PATH = BACKEND_DIR / "agent_state.json"
# BACKEND_DIR이 Backend 폴더를 가리키면, 그 부모가 프로젝트 루트
PROJECT_ROOT = Path(BACKEND_DIR).parent
DEFAULT_TEXT_SINK = str((PROJECT_ROOT / "Frontend" / "public" / "longText.txt").resolve())
DEFAULT_OUTPUT_PATH = "Frontend/public/longText.txt"   # 응답란
DEFAULT_SEPARATE_DIR = "Output"                        # 별도 저장 기본 폴더 (프로젝트 루트 기준)
PLAIN_MAX_CHAT_CHARS = 1800
CODE_MAX_CHAT_LINES = 80
try:
    DEFAULT_TEXT_SINK  # 이미 있으면 재정의 안 함
except NameError:
    try:
        # FRONTEND_DIR이 있다면 그걸 우선 사용
        DEFAULT_TEXT_SINK = str((FRONTEND_DIR / "public" / "longText.txt").resolve())
    except Exception:
        # 없다면 BACKEND_DIR 기준으로 프로젝트 루트를 거슬러 올라가 계산
        from pathlib import Path
        PROJECT_ROOT = Path(BACKEND_DIR).parent
        DEFAULT_TEXT_SINK = str((PROJECT_ROOT / "Frontend" / "public" / "longText.txt").resolve())

# [추가] 모델 우선순위를 로드하는 중앙 함수
def load_model_priority():
    """
    agent_state.json 파일에서 모델 우선순위 목록을 로드합니다.
    파일이 없거나 문제가 있을 경우 기본 목록을 반환합니다.
    """
    # 기본값: 가장 일반적인 고성능 모델 목록
    default_models = ['gemini-1.5-pro-latest', 'gemini-1.5-flash-latest']
    
    if not AGENT_STATE_PATH.exists():
        print("⚠️ agent_state.json 파일을 찾을 수 없어 기본 모델 목록을 사용합니다.")
        return default_models

    try:
        with open(AGENT_STATE_PATH, 'r', encoding='utf-8') as f:
            state = json.load(f)
            model_list = state.get("model_priority_list")
            if model_list and isinstance(model_list, list) and len(model_list) > 0:
                print(f"💡 중앙 설정에서 모델 우선순위를 로드했습니다: {model_list}")
                return model_list
            else:
                print("⚠️ agent_state.json에 유효한 모델 목록이 없어 기본값을 사용합니다.")
                return default_models
    except Exception as e:
        print(f"⚠️ agent_state.json 로딩 중 오류 발생, 기본 모델 목록을 사용합니다: {e}")
        return default_models

def dict_to_schema(schema_dict: dict) -> types.Schema | None:
    """
    JSON 파일에서 읽어온 파라미터 딕셔너리를 재귀적으로 types.Schema 객체로 변환합니다.
    """
    if not schema_dict or not isinstance(schema_dict, dict): return None
    type_enum = getattr(types.Type, schema_dict.get("type", "TYPE_UNSPECIFIED").upper(), types.Type.TYPE_UNSPECIFIED)
    properties = {k: dict_to_schema(v) for k, v in schema_dict.get("properties", {}).items()} if "properties" in schema_dict else None
    items = dict_to_schema(schema_dict["items"]) if "items" in schema_dict else None
    return types.Schema(type=type_enum, properties=properties, items=items, required=schema_dict.get("required"), description=schema_dict.get("description"))
def schema_to_dict(schema: types.Schema) -> dict:
    if not schema:
        return {}
    out = {}
    # type
    t = getattr(schema, "type", None)
    if t and getattr(t, "name", None):
        type_str = t.name.lower()
        if type_str not in {"object","array","string","number","integer","boolean"}:
            type_str = None
    else:
        type_str = None
    # infer when missing
    if not type_str:
        if getattr(schema, "properties", None):
            type_str = "object"
        elif getattr(schema, "items", None):
            type_str = "array"
    if type_str:
        out["type"] = type_str

    # simple fields
    if getattr(schema, "format", None): out["format"] = schema.format
    if getattr(schema, "description", None): out["description"] = schema.description
    if getattr(schema, "nullable", None): out["nullable"] = schema.nullable
    if getattr(schema, "enum", None): out["enum"] = list(schema.enum)

    # nested
    if getattr(schema, "items", None):
        out["items"] = schema_to_dict(schema.items)
    if getattr(schema, "properties", None):
        out["properties"] = {k: schema_to_dict(v) for k, v in schema.properties.items()}

    # required (filter unknowns)
    req = getattr(schema, "required", None)
    if req:
        props = set(out.get("properties", {}).keys())
        filtered = [k for k in req if k in props] if props else list(req)
        if filtered:
            out["required"] = filtered
    if _PB_to_dict is not None and (not out or not out.get("properties")):
        try:
            pb = getattr(schema, "_pb", None) or schema
            out_pb = _PB_to_dict(pb, preserving_proto_field_name=True) or {}
            if isinstance(out_pb, dict) and out_pb:
                return out_pb
        except Exception:
            pass
    return out
def function_declaration_to_dict(d: types.FunctionDeclaration) -> dict:
    """
    [핵심 수정] FunctionDeclaration 객체를 ChromaDB 메타데이터 형식으로 변환합니다.
    도구의 전체 명세를 'definition_json' 키에 JSON 문자열로 저장합니다.
    """
    # 1. FunctionDeclaration 객체를 완전한 딕셔너리로 변환합니다.
    declaration_dict = {
        "name": d.name,
        "description": d.description,
        "parameters": schema_to_dict(d.parameters) if d.parameters else {}
    }

    # 2. ChromaDB에 저장할 최종 메타데이터를 구성합니다.
    return {
        "name": d.name,
        "description": d.description,
        # 3. 전체 명세 딕셔너리를 JSON 문자열로 만들어 저장합니다.
        #    이것이 agent.py에서 명세를 올바르게 불러오는 열쇠입니다.
        "definition_json": json.dumps(declaration_dict, ensure_ascii=False)
    }
def sync_tools_with_vector_db(definitions: list[types.FunctionDeclaration], collection: chromadb.Collection, model: SentenceTransformer, tool_meta: dict | None = None):

    """
    메모리에 로드된 모든 도구 정의를 VectorDB와 동기화합니다.
    서버 시작 시 호출되어 VectorDB가 항상 최신 상태를 유지하도록 보장합니다.
    """
    print("🔄 Synchronizing tools with VectorDB...")
    try:
        existing_tool_ids = set(collection.get(include=[])['ids'])
        print(f"[VectorDB] existing ids: {sorted(list(existing_tool_ids))}")
        print(f"[VectorDB] loaded defs: {[d.name for d in definitions]}")
    except Exception as e:
        print(f"⚠️ VectorDB에서 기존 도구 ID를 가져오는 데 실패했습니다: {e}")
        existing_tool_ids = set()

    definitions_to_add = [
        definition for definition in definitions if definition.name not in existing_tool_ids
    ]

    if definitions_to_add:
        ids = [d.name for d in definitions_to_add]
        # 검색의 기반이 되는 문서는 도구의 '설명'입니다.
        documents = [d.description for d in definitions_to_add]
        def _with_meta(fd, tool_meta):
            meta = function_declaration_to_dict(fd)  # 표준 정의 -> dict
            m = (tool_meta or {}).get(fd.name, {}) or {}
            if m.get("tags"):
                meta["tags"] = m["tags"]
            dg = m.get("dependency_graph") or {}
            if isinstance(dg, dict):
                nodes = dg.get("nodes") or []
                covers = [n.get("tool_name") for n in nodes if isinstance(n, dict) and n.get("tool_name")]
                if covers:
                    meta["covers"] = covers
            return meta

        metadatas = [_with_meta(d, tool_meta) for d in definitions_to_add]
        
        print(f"  - Adding {len(definitions_to_add)} tools to VectorDB: {ids}")
        
        embeddings = model.encode(documents).tolist()
        
        collection.add(embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids)
        print(f"✅ Successfully added {len(definitions_to_add)} tools to VectorDB.")
    else:
        print("✅ VectorDB is already up-to-date with all loaded tools.")
def _parse_definitions_blob(raw_text: str):
    """
    raw_text를 JSON/JSON5/JSONL/혼합 리스트 등 다양한 포맷에서
    [ { ..tool dict.. }, ... ] 형태의 리스트로 정규화한다.
    """
    def _as_list(x):
        # dict 한 개면 리스트로 감싸기
        if isinstance(x, dict):
            # {"tools":[...]} 형태면 그대로 tools
            if "tools" in x and isinstance(x["tools"], list):
                return [i for i in x["tools"] if isinstance(i, (dict, str))]
            return [x]
        if isinstance(x, list):
            return [i for i in x if isinstance(i, (dict, str))]
        return []

    # 1) 우선 JSON5로 통짜 파싱을 시도
    try:
        parsed = json5.loads(raw_text)
        items = _as_list(parsed)
        # 리스트 안의 문자열 요소는 개별적으로 다시 파싱
        out = []
        for item in items:
            if isinstance(item, str):
                try:
                    out.append(json5.loads(item))
                except Exception:
                    continue
            elif isinstance(item, dict):
                out.append(item)
        if out:
            return out
    except Exception:
        pass

    # 2) JSON Lines 시도 (줄마다 객체)
    out = []
    for ln in raw_text.splitlines():
        ln = ln.strip()
        if not ln or not ln.startswith("{") or not ln.endswith("}"):
            continue
        try:
            obj = json5.loads(ln)
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:
            continue
    if out:
        return out

    # 3) 실패하면 빈 리스트
    return []
class ToolManager:
    def __init__(self):
        self.action_handlers = {}
        self.all_definitions = []
        self.defs_index = {}      # ← 나중에 reload()에서 빌드
        self.tool_meta = {}       # ← _load_generated_skills()에서 채움(이미 그렇게 해놨다면 유지)
        self.tools = []
        self.reload()
    @property
    def tool_catalog(self) -> Dict[str, Dict[str, Any]]:
        """
        이름 -> 스펙(dict) 카탈로그를 즉시 구성해 돌려준다.
        - source: self.all_definitions (FunctionDeclaration)
        - meta  : self.tool_meta (tags, dependency_graph)
        - params: schema_to_dict()로 파라미터 스키마를 dict로 변환
        """
        catalog: Dict[str, Dict[str, Any]] = {}
        for fd in self.all_definitions:
            name = getattr(fd, "name", None)
            if not name:
                continue
            spec: Dict[str, Any] = {
                "name": name,
                "description": getattr(fd, "description", "") or "",
                "parameters": schema_to_dict(getattr(fd, "parameters", None)) or {},
            }
            meta = self.tool_meta.get(name, {}) or {}
            if meta.get("tags"):
                spec["tags"] = list(meta["tags"])
            if meta.get("dependency_graph"):
                spec["dependency_graph"] = meta["dependency_graph"]
            catalog[name] = spec
        return catalog
    def ensure_args(self, tool_name: str, base_args: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """
        에이전트에서 기대하는 트리플 (ok, normalized, err) 형태로 파라미터 정규화.
        - self.defs_index[tool_name] 안의 allowed/required를 사용
        - DEFAULT_OUTPUT_PATH 주입(허용 키에 있고 미지정인 경우)
        - 필수키 누락 검사 및 불필요 키 제거
        """
        try:
            idx = (self.defs_index or {}).get(tool_name) or {}
            allowed: set = set(idx.get("allowed") or [])
            required: set = set(idx.get("required") or [])
        except Exception:
            return False, {}, f"tool '{tool_name}' schema not indexed"

        args = dict(base_args or {})

        # 기본값 보정: output_path 허용 시 기본 싱크 주입
        try:
            if ("output_path" in allowed) and ("output_path" not in args):
                try:
                    from tool_registry import DEFAULT_OUTPUT_PATH
                except Exception:
                    DEFAULT_OUTPUT_PATH = "Frontend/public/longText.txt"
                args["output_path"] = DEFAULT_OUTPUT_PATH
        except Exception:
            pass

        # (선택) topic 등의 파생 필드 보정이 필요하면 여기서 처리
        if ("topic" in required) and ("topic" not in args):
            title = args.get("title")
            if title:
                args["topic"] = title

        # 필수 충족 확인
        if not required.issubset(set(args.keys())):
            missing = sorted(list(required.difference(args.keys())))
            return False, {}, f"missing required: {missing}"

        # 불필요 키 제거
        filtered = {k: v for k, v in args.items() if (k in allowed) or (k in required)}

        return True, filtered, None
    def reload(self):
        print("🔄 [System] 모든 스킬을 메모리로 리로드합니다...")

        self.action_handlers.clear()
        self.all_definitions.clear()

        from tools.basic_handlers import basic_action_handlers
        from tools.basic_definitions import basic_tool_definitions

        # 1) 기본 도구 로드
        self.action_handlers.update(basic_action_handlers)
        self.all_definitions.extend(basic_tool_definitions)

        # 2) 생성 도구 로드 (여기서 self.tool_meta도 채워진 상태여야 함)
        generated_handlers, generated_definitions_obj = self._load_generated_skills()
        self.action_handlers.update(generated_handlers)
        self.all_definitions.extend(generated_definitions_obj)

        # 3) defs_index 빌드 (모든 정의를 모은 뒤)
        self.defs_index = {
            getattr(fd, "name"): {
                "allowed": set(schema_to_dict(getattr(fd, "parameters", {})).get("properties", {}).keys()),
                "required": set(schema_to_dict(getattr(fd, "parameters", {})).get("required", []) or []),
            }
            for fd in self.all_definitions
            if getattr(fd, "name", None)
        }

        # 4) 최종 Tool 묶음 구성 (FunctionDeclaration만 넣기 — tags/dep는 넣지 않음)
        self.tools = [types.Tool(function_declarations=self.all_definitions)]
        # --- add: build in-memory tool_meta from loaded definitions ---
        self.tool_meta = {}
        for d in self.all_definitions:
            name = getattr(d, "name", None)
            if not name:
                continue
            meta_entry = {}
            # 1) dependency_graph: definitions.json 항목에 있든(추가필드), 기존 materialize 병합이든
            dep = getattr(d, "dependency_graph", None)
            if dep:
                meta_entry["dependency_graph"] = dep
            # 2) tags: FunctionDeclaration에 공식 필드는 없지만, 우리가 getattr로 붙여둔 값이 있으면 흡수
            tg = getattr(d, "tags", None) or getattr(d, "tool_tags", None)
            if tg:
                meta_entry["tags"] = tg
            if meta_entry:
                self.tool_meta[name] = meta_entry
    def _load_generated_skills(self):
        """
        AI가 생성한 스킬들을 파일에서 동적으로 불러옵니다.
        """
        generated_handlers = {}
        generated_definitions_obj = []
        tool_meta = {}

        # 1. 생성된 JSON 정의 파일 로딩
        if GENERATED_DEFINITIONS_PATH.exists() and GENERATED_DEFINITIONS_PATH.stat().st_size > 0:
            with open(GENERATED_DEFINITIONS_PATH, 'r', encoding='utf-8') as f:
                defs_raw = json5.load(f)

            # 1) 래핑/형식 정규화
            if isinstance(defs_raw, dict) and "tools" in defs_raw:
                definitions_list = defs_raw.get("tools", [])
            elif isinstance(defs_raw, list):
                definitions_list = defs_raw
            else:
                definitions_list = []

            # 2) materialized_tools.json에서 dependency_graph 병합
            try:
                mat_path = os.path.join(BACKEND_DIR, "state", "materialized_tools.json")
                name_to_dep = {}
                if os.path.exists(mat_path) and os.path.getsize(mat_path) > 0:
                    mat = json5.load(open(mat_path, "r", encoding="utf-8"))
                    for meta in (mat or {}).values():
                        d = (meta or {}).get("meta", {}).get("definition", {}) or {}
                        nm = d.get("new_tool_name") or d.get("name")
                        dep = d.get("dependency_graph")
                        if nm and dep:
                            name_to_dep[nm] = dep
            except Exception as _merge_e:
                print(f"[WARN] materialized dependency_graph merge skipped: {_merge_e}")

            # 3) FunctionDeclaration 생성(+ tags/dep_graph 주입)
            # 3) FunctionDeclaration 생성(표준 키만) + side-meta 저장
            for definition_data in definitions_list:
                if not isinstance(definition_data, dict):
                    continue
                try:
                    name = definition_data.get("new_tool_name") or definition_data.get("name")
                    description = definition_data.get("description") or ""
                    parameters_schema = dict_to_schema(definition_data.get("parameters", {}))
                    if not (name and description):
                        continue

                    fd = types.FunctionDeclaration(
                        name=name,
                        description=description,
                        parameters=parameters_schema
                    )
                    generated_definitions_obj.append(fd)

                    # ← FunctionDeclaration에는 tags/dep 주입 금지. meta로만 보관.
                    tool_meta[name] = {
                        "tags": definition_data.get("tags") or definition_data.get("tool_tags") or [],
                        "dependency_graph": definition_data.get("dependency_graph") or name_to_dep.get(name) or {},
                    }

                except Exception as e:
                    nm = (definition_data.get("new_tool_name") or definition_data.get("name") or "<unknown>")
                    print(f"⚠️ 경고: 생성된 스킬 '{nm}' 정의 로딩 중 오류: {e}")
        # 2. 생성된 Python 코드 파일 로드 및 함수 추출
        if GENERATED_SKILLS_PATH.exists() and GENERATED_SKILLS_PATH.stat().st_size > 0:
            try:
                with open(GENERATED_SKILLS_PATH, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                
                module_namespace = {}
                # --- Inject StrictActionHandlers once, right before exec ---
                import builtins, inspect, functools

                # (1) defs_index 준비: basic + 이번에 읽은 generated definitions 결합
                def _schema_to_dict(schema):
                    # 이미 config.py에 있는 schema_to_dict 써도 되면 그걸 import해서 써도 OK
                    try:
                        from config import schema_to_dict as _schema_to_dict_shared
                        return _schema_to_dict_shared(schema)
                    except Exception:
                        # fallback: 아주 단순한 형태
                        d = {}
                        if getattr(schema, "properties", None):
                            d["properties"] = {k: {} for k in schema.properties.keys()}
                        if getattr(schema, "required", None):
                            d["required"] = list(schema.required)
                        return d

                combined_defs = list(self.all_definitions) + list(generated_definitions_obj)
                defs_index_for_wrapper = {}
                for fd in combined_defs:
                    name = getattr(fd, "name", None)
                    if not name:
                        continue
                    sd = _schema_to_dict(getattr(fd, "parameters", {})) or {}
                    props = (sd.get("properties") or {})
                    req   = (sd.get("required") or [])
                    defs_index_for_wrapper[name] = {
                        "allowed": set(props.keys()),
                        "required": set(req),
                    }

                # (2) StrictActionHandlers: 스키마 검증 + 핸들러 시그니처 어댑트
                class _StrictActionHandlers(dict):
                    def __init__(self, handlers, defs_index):
                        super().__init__(handlers)
                        self._defs_index = defs_index

                    def __getitem__(self, tool_name):
                        fn = super().__getitem__(tool_name)
                        spec = self._defs_index.get(tool_name, {}) or {}
                        allowed = spec.get("allowed", set())
                        required = spec.get("required", set())

                        sig = inspect.signature(fn)
                        params = list(sig.parameters.values())
                        one_arg = (len(params) == 1)  # handle({dict}) 스타일 핸들러 호환

                        async def _call_wrapper(**kwargs):
                            # 1) 스키마 키 검증
                            keys = set(kwargs.keys())
                            unknown = keys - allowed
                            missing = required - keys
                            if unknown:
                                raise TypeError(
                                    f"핸들러 인자 불일치로 실행 실패: {tool_name}에 정의 밖 키 존재: {sorted(list(unknown))}"
                                )
                            if missing:
                                raise TypeError(
                                    f"핸들러 인자 불일치로 실행 실패: {tool_name}에 필수 키 누락: {sorted(list(missing))}"
                                )

                            # 2) 시그니처 어댑트
                            if one_arg:
                                # 단일 인자(action_input 한 덩어리)인 핸들러는 dict를 포지셔널로
                                return await fn(kwargs)
                            else:
                                # 키워드 인자를 받는 핸들러는 그대로 키워드로
                                return await fn(**kwargs)

                        return _call_wrapper

                # (3) exec 네임스페이스 + 내장에 주입
                module_namespace.setdefault("__builtins__", __builtins__)
                module_namespace["StrictActionHandlers"] = _StrictActionHandlers
                builtins.StrictActionHandlers = _StrictActionHandlers

                # Normalize namespace
                if not isinstance(module_namespace, dict):
                    module_namespace = {}
                module_namespace.setdefault("__builtins__", __builtins__)
                # Provide symbol to both exec namespace and builtins (for type-hint resolution)
                module_namespace["StrictActionHandlers"] = _StrictActionHandlers
                builtins.StrictActionHandlers = _StrictActionHandlers

                # (Optional) make type hints lazy so name lookup isn’t needed at import time
                if "from __future__ import annotations" not in file_content.splitlines()[:3]:
                    file_content = "from __future__ import annotations\n" + file_content

                exec(file_content, module_namespace)
                
                for name, func in module_namespace.items():
                    if inspect.iscoroutinefunction(func):
                        wrapped_func = functools.partial(
                            func, StrictActionHandlers(self.action_handlers, defs_index_for_wrapper)
                        )
                        generated_handlers[name] = wrapped_func
            except Exception as e:
                name_for_log = None
                if isinstance(definition_data, dict):
                    name_for_log = definition_data.get("new_tool_name") or definition_data.get("name")
                print(f"⚠️ 경고: 생성된 스킬 '{name_for_log or '<unknown>'}' 정의 로딩 중 오류: {e}")
        self.tool_meta = tool_meta
        return generated_handlers, generated_definitions_obj
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
vector_db_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
tool_collection = vector_db_client.get_or_create_collection(name="kaede_tools", metadata={"hnsw:space": "cosine"})
# 앱 전체에서 공유할 단일 인스턴스
tool_manager = ToolManager()
async def handle_reload_skills(action_input: dict):
    """[시스템] 모든 스킬을 메모리로 다시 로드합니다."""
    try:
        tool_manager.reload()
        return {"status": "success", "message": "모든 스킬을 성공적으로 리로드했습니다."}
    except Exception as e:
        return {"status": "error", "reason": str(e)}
async def stream_text_by_char(stream_type: str, text: str, delay: float = 0.01):
    """텍스트를 한 글자씩 스트리밍하는 제너레이터 헬퍼 함수"""
    for char in text:
        yield f"data: {json.dumps({'type': stream_type, 'char': char})}\n\n"
        await asyncio.sleep(delay)
def safe_json_parse(s):
    try:
        return json5.loads(s)
    except Exception as e:
        print(f"[json5 파싱 실패] {e}")
        return None
class StrictActionHandlers(dict):
    """
    action_handlers에 대한 엄격 프록시.
    - 내부 도구 호출 시 전달된 kwargs에서 '허용되지 않은 키'를 자동 제거하여 크래시를 방지한다.
    - 허용 키 판정은 실행 대상 함수의 실제 시그니처를 inspect하여 수행한다.
    - 함수가 **kwargs를 받으면 그대로 모두 통과시킨다.
    - 함수가 비동기(async)거나 동기냐를 자동 판별해 적절히 호출한다.
    """
    def __init__(self, backing: dict):
        super().__init__()
        self._backing = backing
        self._sig_cache = {}

    def __getitem__(self, name: str):
        fn = self._backing[name]

        async def _acall_filtered(**kwargs):
            filtered = self._filter_kwargs(fn, kwargs)
            if asyncio.iscoroutinefunction(fn):
                return await fn(**filtered)
            return fn(**filtered)

        def _call_filtered(**kwargs):
            filtered = self._filter_kwargs(fn, kwargs)
            return fn(**filtered)

        # 항상 async로 래핑하면 호출 측 await 패턴을 안 건드려도 됨
        return _acall_filtered

    def _filter_kwargs(self, fn, kwargs: dict) -> dict:
        try:
            sig = self._sig_cache.get(fn)
            if sig is None:
                sig = inspect.signature(fn)
                self._sig_cache[fn] = sig

            # **kwargs 허용이면 필터링 불필요
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                return kwargs

            # 키워드 인자만 허용 목록으로
            allowed = {
                name for name, p in sig.parameters.items()
                if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            }
            filtered = {k: v for k, v in kwargs.items() if k in allowed}

            if filtered.keys() != kwargs.keys():
                dropped = [k for k in kwargs.keys() if k not in filtered]
                print(f"[StrictActionHandlers] Dropped unknown keys for '{getattr(fn,'__name__',str(fn))}': {dropped}")

            return filtered
        except Exception as e:
            # 문제 발생 시 안전하게 원본 그대로 반환 (최소한 동작 유지)
            print(f"[StrictActionHandlers] Filter error: {e}")
            return kwargs