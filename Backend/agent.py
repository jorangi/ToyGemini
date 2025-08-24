import asyncio
import random
from pathlib import Path
import json
import json5
import uuid, datetime
from google.genai import types
from llm_utils import call_gemini_agent
from prompt_builder import build_prompt, extract_json_from_text
from sql_router import get_db_schema, Session, User, SessionLocal, ToolCallLog, save_message
from metatools.handlers import run_optimization_workflow, generate_tool_definition_and_code, register_newly_generated_tool, handle_reload_skills
from config import tool_manager, stream_text_by_char, safe_json_parse, MAX_AGENT_ITERATIONS, embedding_model, tool_collection, AGENT_STATE_PATH, load_model_priority
from optimization_manager import OptimizationManager
from tool_registry import ToolRegistry
from metatools.tool_selector import decide_tool, propose_parameters
from config import BACKEND_DIR
import re
from metatools.tool_selector import classify_goal_with_llm
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Deque, Dict, Any
from collections import deque
from pathlib import Path as _P
import os, json, re, hashlib, time
import inspect
import re, os, hashlib
from pathlib import Path
from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, Any
_PATH_RX = re.compile(
    r"""
    (                                   # Windows 절대경로(C:\...) 또는 POSIX(/... , ./... , ../...)
        [A-Za-z]:[\\/](?:[^\\/:*?"<>|\r\n]+[\\/]?)+
        |
        (?:\.\.?[/\\])?(?:[/\\][^\\/:*?"<>|\r\n]+)+
    )
    """,
    re.VERBOSE,
)
from dataclasses import dataclass
from metatools.tool_selector import select_best_covering_tool

@dataclass
class _Effect:
    kind: str           # 'file_mutation' | 'file_backup' | 'unknown'
    ok: bool
    primary_path: str = ""
    aux_path: str = ""
    meta: dict | None = None

def _adapt_action_params_for(handler, action: str, params: dict) -> dict:
        """
        STRICT MODE:
        - 핸들러 시그니처에 정의된 키만 통과
        - 동의어/별칭/자동치환 금지
        - **kwargs 허용 핸들러는 그대로 통과
        """
        if not isinstance(params, dict):
            return {}
        try:
            sig = inspect.signature(handler)
        except Exception:
            # 시그니처를 못 읽으면 있는 그대로 전달 (테스트 호환)
            return params

        # **kwargs 허용이면 필터링하지 않음
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return params

        expected = set(sig.parameters.keys())
        # 정확히 일치하는 키만 통과
        return {k: v for k, v in params.items() if k in expected}

class Agent:
    def __init__(self, user_goal: str, session_id: str, background_tasks, model_priority_list: list):
        self.tool_manager = tool_manager
        self.user_goal = user_goal
        self.session_id = session_id
        self.background_tasks = background_tasks
        self.available_models = load_model_priority()
        self._backed_up_paths = set()
        self._written_paths = set()   # ← 새로 추가: 이번 세션에서 '실제 쓰기'가 발생한 경로 추적
        self.request_id = str(uuid.uuid4())
        self.db = SessionLocal()
        self.prompt_content = []
        self.session_workflow = []
        self.last_written_file = None
        self.current_user_id = None
        self._backed_up_paths = set()
        self.tool_definitions = self._load_tool_definitions()
        self.optim_manager = OptimizationManager()
        self._materialize_queue = []
        self._effects: Deque[_Effect] = deque(maxlen=8)
        self._file_snapshots: Dict[str, tuple] = {}
        try:
            if AGENT_STATE_PATH.exists():
                with open(AGENT_STATE_PATH, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    self.available_models = state.get("model_priority_list", list(model_priority_list))
                    print(f"💡 저장된 모델 우선순위를 로드했습니다: {self.available_models}")
            else:
                self.available_models = list(model_priority_list)
        except Exception as e:
            print(f"⚠️ Agent 상태 로딩 실패, 기본값 사용: {e}")
            self.available_models = list(model_priority_list)
    # ==== 넣을 위치: Agent 클래스 내부(중복 정의 정리) ====
    from dataclasses import dataclass
    from pathlib import Path as _P
    import hashlib, os, time, re
    # --- Agent 클래스 내부: 단일 진입 호출 래퍼 ---
    # --- Agent._call_tool: observation_result는 여기서만 만든다 ---
    import inspect, time, json
    def _user_explicitly_requested_path(self, candidate: str) -> bool:
        """
        사용자가 '문자 그대로' 파일 경로/파일명을 요청에 포함했을 때만 True.
        - 절대경로이거나 디렉터리 구분자 포함 → 명시로 간주
        - 또는 user_goal 내에 정확히 동일한 토큰이 존재
        """
        if not candidate or not isinstance(candidate, str):
            return False
        if os.path.isabs(candidate) or (os.sep in candidate or "/" in candidate):
            return True
        goal = (self.user_goal or "")
        # 공백/구두점 경계에서 정확히 일치하는지 확인
        pat = r'(?<!\w)' + re.escape(candidate) + r'(?!\w)'
        return re.search(pat, goal) is not None
    def _get_planned_steps(self) -> list[str]:
        """
        최신 플랜에서 steps(action 이름)만 뽑아온다.
        LLM이 낸 계획이 없으면, 현재 액션만 단일 스텝으로 간주.
        """
        plan_obj = getattr(self, "_latest_plan_obj", None) or {}
        steps = []
        for s in (plan_obj.get("steps") or []):
            # step: {"action":"write_file", "parameters":{...}}
            a = (s or {}).get("action")
            if a:
                steps.append(a)
        return steps
    async def _call_tool(self, action: str, action_input: dict) -> dict:
        handler = getattr(self.tool_manager, "action_handlers", {}).get(action)
        if not handler:
            return {"status": "error", "reason": f"알 수 없는 Agent 액션: '{action}'."}

        # (선택) 입력 스키마 검증
        if hasattr(self, "_validate_action_input") and hasattr(self, "tool_definitions"):
            try:
                ok, schema_err = self._validate_action_input(action, action_input or {})
                if not ok:
                    # schema_err 자체가 observation_result
                    print(f"[✅ Observation] {json.dumps(schema_err, ensure_ascii=False, indent=2)}")
                    return schema_err
            except Exception as e:
                print(f"[Schema Validate Warning] {e}")

        # 핸들러 시그니처에 맞게 파라미터 정규화
        params = self._normalize_and_filter_params_for_handler(action, action_input or {})

        t0 = time.perf_counter()
        try:
            sig = inspect.signature(handler)
            # 핸들러가 action_input 하나만 받는 경우 지원
            if list(sig.parameters.keys()) == ["action_input"]:
                observation_result = await handler(params if "action_input" not in params else params["action_input"])
            else:
                observation_result = await handler(**params)
        except TypeError as e:
            # 인자 불일치
            try:
                sig_str = str(inspect.signature(handler))
            except Exception:
                sig_str = "(signature unavailable)"
            observation_result = {
                "status": "error",
                "reason": f"핸들러 인자 불일치로 실행 실패: {e}",
                "handler_signature": sig_str,
                "given_params": action_input or {},
            }
        except Exception as e:
            # 실행 중 예외
            observation_result = {"status": "error", "reason": f"핸들러 실행 중 예외: {e}"}

        # 관측 로그 출력
        try:
            print(f"[✅ Observation] {json.dumps(observation_result, ensure_ascii=False, indent=2)}")
        except Exception:
            print(f"[✅ Observation] {observation_result}")

        # 사후 처리(파일 경로 추적/보강 훅 등) — 여기서 observation_result만 넘김
        try:
            await self._on_action_observed(action, action_input or {}, observation_result or {})
        except Exception as e:
            print(f"[Post-Observe] error: {e}")

        return observation_result or {}
    @staticmethod
    def _file_sig(path: str) -> tuple[int, int, str]:
        """파일 변경 감지를 위한 가벼운 시그니처."""
        try:
            st = os.stat(path)
            size = st.st_size
            mtime = int(st.st_mtime)
            h = hashlib.sha1()
            with open(path, "rb") as f:
                h.update(f.read(65536))
                if size > 65536:
                    f.seek(max(0, size - 65536))
                    h.update(f.read(65536))
            return (size, mtime, h.hexdigest())
        except Exception:
            return (0, 0, "")
    @staticmethod
    def _is_auto_placeholder(text: str) -> bool:
        return isinstance(text, str) and re.fullmatch(r"\s*<\s*AUTO[^>]*>\s*", text or "") is not None
    @staticmethod
    def _normalize_for_dup(self, s: str) -> str:
        # 문단 중복 제거용 간단 정규화
        return re.sub(r"\s+", " ", re.sub(r"[^\w가-힣]+", "", (s or "").lower())).strip()
    @staticmethod
    def _dedupe_paragraphs(text: str) -> str:
        """완전 동일 문단 반복 방지. 자연어 키워드 불사용."""
        if not isinstance(text, str):
            return text
        paras = re.split(r"(?:\r?\n){2,}", text)
        seen, out = set(), []
        for p in paras:
            key = p.strip()
            if key and key not in seen:
                seen.add(key); out.append(p)
        return "\n\n".join(out)
    @staticmethod
    def _looks_write_success(observation: dict) -> bool:
        if not isinstance(observation, dict) or observation.get("status") != "ok":
            return False
        det = observation.get("detail", {}) if isinstance(observation.get("detail"), dict) else {}
        return bool(observation.get("path") or (isinstance(det, dict) and det.get("path")))
    @staticmethod
    def _looks_backup_success(observation: dict) -> bool:
        if not isinstance(observation, dict) or observation.get("status") != "ok":
            return False
        det = observation.get("detail", {}) if isinstance(observation.get("detail"), dict) else {}
        src = observation.get("src") or (det.get("src") if isinstance(det, dict) else None)
        bak = observation.get("backup_path") or (det.get("backup_path") if isinstance(det, dict) else None)
        return bool(src and bak)
    async def _plan_with_retry(self, user_goal_text: str, max_attempts: int = 2) -> dict | None:
        """
        통합 계획을 LLM으로부터 JSON으로 받아오되, 실패 시 1회 재시도.
        - 1차: 현재 기본 모델/설정
        - 2차: 더 보수적인 설정(온도 0, 다른 우선순위 모델)로 재시도
        반환: 계획 객체(dict) 또는 None
        """
        # 프롬프트는 기존 플래너에서 쓰던 ‘계획 JSON’ 지시문을 그대로 재사용하되,
        # 여기서는 별도 하드코딩 없이, 단지 JSON 파싱 성공 여부만 본다.
        base_instruction = (
            "Produce a valid JSON plan object. Do not include any non-JSON text.\n"
            "The JSON must parse with a standard parser."
        )
        for attempt in range(1, max_attempts + 1):
            try:
                use_tools = False
                available = getattr(self, "available_models", None)
                extra = {}
                if attempt == 2:
                    # 두 번째 시도: 더 보수적인 디코딩을 위해 옵션만 살짝 보수적으로
                    extra["temperature"] = 0
                    # 모델 우선순위를 내부적으로 바꿀 수 있다면 _call_llm_safe가 이를 반영

                prompt = f"{base_instruction}\n\nUSER_GOAL:\n{user_goal_text or ''}"
                text, _ = await self._call_llm_safe(
                    [self._build_user_message(prompt)],
                    available_models=available,
                    use_tools=use_tools,
                    **extra
                )
                plan = json.loads(text.strip())
                if isinstance(plan, dict):
                    return plan
            except Exception:
                continue
        return None
    def _messages_fingerprint(self, messages) -> str:
        """
        LLM 호출 메시지(리스트)를 안정적으로 직렬화해 해시 키 생성.
        role/text/parts만 보존(내용 동일하면 동일 키).
        """
        def norm_msg(m):
            role = ""
            text = ""
            try:
                role = getattr(m, "role", "") or m.get("role", "")
                parts = getattr(m, "parts", None) or m.get("parts", [])
                buf = []
                for p in parts:
                    t = getattr(p, "text", None)
                    if t is None and isinstance(p, dict):
                        t = p.get("text")
                    if isinstance(t, str):
                        buf.append(t)
                text = "\n".join(buf)
            except Exception:
                text = str(m)
            return {"role": role, "text": text}

        try:
            norm = [norm_msg(m) for m in (messages or [])]
            payload = json.dumps(norm, ensure_ascii=False, sort_keys=True)
        except Exception:
            payload = str(messages)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _parse_retry_delay_sec(self, exc) -> int:
        """
        예외 payload에서 'retryDelay': '18s' 같은 값을 추출. 실패하면 기본 5초.
        """
        txt = ""
        try:
            txt = str(getattr(exc, "args", [""])[-1]) or ""
        except Exception:
            pass
        m = re.search(r"retryDelay[^0-9]*([0-9]+)\s*s", txt)
        if m:
            try:
                return max(1, int(m.group(1)))
            except Exception:
                pass
        return 5
    def _extract_written_file_path(self, action: str, action_input: dict, observation: dict) -> str | None:
        det = observation.get("detail", {}) if isinstance(observation.get("detail"), dict) else {}
        p = det.get("path") or observation.get("path")
        if p: return str(p)
        for k in ("file_path", "path", "target_path"):
            if k in (action_input or {}): return str(action_input[k])
        return getattr(self, "last_written_file", None)
    def _normalize_and_filter_params_for_handler(self, action: str, raw: dict) -> dict:
        params = dict(raw or {})
        handler = getattr(self.tool_manager, "action_handlers", {}).get(action)
        if not handler:
            return params
        try:
            sig = inspect.signature(handler)
            expected = set(sig.parameters.keys())
        except Exception:
            return params
        if "action_input" in expected:
            return {"action_input": params}
        return {k: v for k, v in params.items() if k in expected}
    def _normalize_effect(self, *args, **kwargs) -> _Effect:
        """
        호출부 호환: _normalize_effect(observation) 또는 _normalize_effect(action, observation) 둘 다 허용.
        """
        if len(args) == 1 and isinstance(args[0], dict):
            observation = args[0]
        elif len(args) >= 2 and isinstance(args[1], dict):
            observation = args[1]
        else:
            return _Effect()

        ok = (observation.get("status") == "ok") if isinstance(observation, dict) else False
        det = observation.get("detail", {}) if isinstance(observation.get("detail"), dict) else {}
        if isinstance(det, dict):
            ok = ok or (det.get("status") in ("ok", "success") or det.get("success") is True)

        path = (isinstance(det, dict) and det.get("path")) or observation.get("path") or ""
        src  = observation.get("src") or (det.get("src") if isinstance(det, dict) else "")
        bkp  = observation.get("backup_path") or (det.get("backup_path") if isinstance(det, dict) else "")

        # 백업 성공
        if ok and src and bkp:
            return _Effect(kind="file_backup", ok=True, primary_path=str(src), aux_path=str(bkp), meta=observation)

        # 쓰기 성공 + 실제 변경
        if ok and path:
            sig_prev = getattr(self, "_file_snapshots", {}).get(path) if hasattr(self, "_file_snapshots") else None
            sig_now  = self._file_sig(path)
            if not hasattr(self, "_file_snapshots"): self._file_snapshots = {}
            self._file_snapshots[path] = sig_now
            if (sig_prev is None) or (sig_prev != sig_now):
                return _Effect(kind="file_mutation", ok=True, primary_path=str(path), meta=observation)

        return _Effect(kind="unknown", ok=ok, primary_path=str(path), meta=observation)
    async def _on_action_observed(self, action: str, action_input: dict, observation: dict):
        """
        1) write_file 성공 시 self.last_written_file 갱신
        2) backup_file 성공 시 최근 중복 백업 억제(선택적 피벗)
        3) 후속 부스트: _post_write_boost(action, action_input, observation)
        """
        try:
            det = observation.get("detail", {}) if isinstance(observation.get("detail"), dict) else {}
            # (1) 최근 작성 파일 경로 기억
            if self._looks_write_success(observation):
                self.last_written_file = (det.get("path") or observation.get("path") or "") or self.last_written_file

            # (2) 백업 성공 시, 너무 촘촘한 중복 백업이면 매크로로 피벗 후보
            if self._looks_backup_success(observation):
                primary = observation.get("src") or det.get("src")
                backup  = observation.get("backup_path") or det.get("backup_path")
                if primary and backup:
                    now = time.time()
                    last = getattr(self, "_last_backup_ts", 0.0)
                    if now - last < 0.8:
                        # 직후 단계에서 매크로로 합치기 시도 (선택)
                        try:
                            action, action_input, meta = self._maybe_override_with_registry_tool(action, action_input)
                            if meta.get("overridden"):
                                print(f"[Registry Override] {meta}")
                        except Exception as _e:
                            print(f"[BackupThrottle Warning] {str(_e)}")
                    self._last_backup_ts = now

            # (3) 후속 부스트 — 잘못된 이중 self 전달 제거
            await self._post_write_boost(action, action_input, observation)
        except Exception as e:
            print(f"[Post-Observe] error: {e}")
    # --- BEGIN: Back-compat shim for tests expecting `_adapt_action_params_for` ---
    import inspect  # 상단에 이미 있다면 중복 import 무관
    if not callable(globals().get("_adapt_action_params_for", None)):
        def _adapt_action_params_for(handler, action: str, params: dict) -> dict:
            """
            STRICT MODE:
            - 핸들러 시그니처에 정의된 키만 통과
            - 동의어/별칭/자동치환 금지
            - **kwargs 허용 핸들러는 그대로 통과
            """
            if not isinstance(params, dict):
                return {}
            try:
                sig = inspect.signature(handler)
            except Exception:
                # 시그니처를 못 읽으면 있는 그대로 전달 (보수적 호환)
                return params
            # **kwargs 허용이면 필터링하지 않음
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                return params
            expected = set(sig.parameters.keys())
            # 정확히 일치하는 키만 통과
            return {k: v for k, v in params.items() if k in expected}
    # --- END: Back-compat shim ---
    async def _llm_decide_destination(self) -> str:
        """
        LLM에게 '출력 목적지가 무엇인지'를 JSON으로 판단받는다.
        - 반환: "inline" | "file"
        - 규칙: 사용자가 '별도의 파일로 저장'을 명시하지 않으면 "inline"
        - 언어/표현 방식에 상관없이 LLM이 의미로 판단 (키워드 리스트 없음)
        """
        user_goal = self.user_goal or ""
        instruction = (
            "You are a strict JSON decision maker.\n"
            "Task: Determine the destination for result delivery.\n"
            "Rules:\n"
            "1) If the user explicitly requests to save as a separate file (persistent artifact), destination = \"file\".\n"
            "2) Otherwise, destination = \"inline\".\n"
            "3) Do not infer intent from vague hints; only explicit file-saving requests count as \"file\".\n"
            "Return JSON only with key: {\"destination\": \"inline\" | \"file\"}."
        )
        prompt = f"{instruction}\n\nUSER_REQUEST:\n{user_goal}"
        try:
            response_text, _ = await self._call_llm_safe(
                [self._build_user_message(prompt)],
                available_models=getattr(self, "available_models", None),
                use_tools=False
            )
            data = json.loads(response_text.strip())
            dest = data.get("destination", "").strip().lower()
            if dest in ("inline", "file"):
                return dest
            return "inline"
        except Exception:
            return "inline"

    async def _llm_author_parameters(self, *, tool_name: str, tool_spec: dict, destination: str, proposed_params: dict) -> dict:
        """
        LLM에게 '선택된 도구의 파라미터'를 최종 작성시키는 단계.
        - tool_spec: 선택된 도구의 전체 스키마(JSON). (도구명/파라미터명 하드코딩 없음)
        - destination: "inline" 또는 "file"
        - proposed_params: 플래너가 제안한 파라미터(초안). LLM이 참고하여 정제.

        규칙(문서화만; 코드로 특정 키 찾지 않음):
        - destination == inline: 별도 파일 경로 지정하지 말고, 기본 싱크를 사용하도록 설계하라.
        - destination == file: 사용자가 명시한 저장 경로가 없다면, 합리적인 기본 산출물 경로를 제안해도 된다.

        반환: 해당 도구 스키마에 맞는 '최종 파라미터' JSON 딕셔너리
        """
        # 기본 싱크 경로 상수는 설정에서 가져와 LLM에 '설명'으로만 전달
        try:
            from tool_registry import DEFAULT_OUTPUT_PATH
        except Exception:
            DEFAULT_OUTPUT_PATH = "Frontend/public/longText.txt"

        # LLM에 제공할 컨텍스트
        tool_schema_json = json.dumps(tool_spec or {}, ensure_ascii=False, indent=2)
        proposed_json = json.dumps(proposed_params or {}, ensure_ascii=False, indent=2)

        instruction = (
            "You are a strict JSON parameter author for a selected tool.\n"
            "You will be given:\n"
            " - TOOL_SCHEMA: the selected tool's full JSON schema (including parameters).\n"
            " - DESTINATION: either \"inline\" or \"file\".\n"
            " - DEFAULT_SINK: the default sink path for inline delivery.\n"
            " - PROPOSED_PARAMS: a draft parameter object from a planner.\n"
            "\n"
            "Your job:\n"
            " - Produce a minimal JSON object that matches the TOOL_SCHEMA.parameters shape.\n"
            " - If DESTINATION == \"inline\": avoid proposing any separate artifact path; instead, rely on the tool's defaults to use DEFAULT_SINK.\n"
            " - If DESTINATION == \"file\": you may include an explicit path only if the user's request clearly specified it; otherwise propose a reasonable output path.\n"
            " - Never add fields not present in TOOL_SCHEMA.parameters.properties.\n"
            " - Preserve essential content from PROPOSED_PARAMS if present (e.g., the main body of text/code), but do not invent file names unless DESTINATION == \"file\".\n"
            " - Return JSON only, no prose.\n"
        )

        user_goal = self.user_goal or ""
        prompt = (
            f"{instruction}\n\n"
            f"USER_REQUEST:\n{user_goal}\n\n"
            f"DESTINATION:\n{destination}\n\n"
            f"DEFAULT_SINK:\n{DEFAULT_OUTPUT_PATH}\n\n"
            f"TOOL_SCHEMA:\n{tool_schema_json}\n\n"
            f"PROPOSED_PARAMS:\n{proposed_json}\n"
        )

        try:
            response_text, _ = await self._call_llm_safe(
                [self._build_user_message(prompt)],
                available_models=getattr(self, "available_models", None),
                use_tools=False
            )
            params = json.loads(response_text.strip())
            if isinstance(params, dict):
                return params
        except Exception:
            pass

        # LLM 실패 시, 최소 파라미터 폴백: 내용만 보존 (경로 관련 입력은 포함하지 않음)
        minimal = {}
        # proposed_params 안에 '내용에 해당하는 값'이 있을 수 있으니, 가장 큰 텍스트 값 하나를 잡아본다(키 이름 가정 없이).
        try:
            longest_text_key = None
            longest_len = -1
            for k, v in (proposed_params or {}).items():
                if isinstance(v, str) and len(v) > longest_len:
                    longest_len = len(v)
                    longest_text_key = k
            if longest_text_key is not None:
                minimal[longest_text_key] = proposed_params[longest_text_key]
        except Exception:
            pass
        return minimal
    async def _dispatch_action(self, action: str, action_input: dict):
        """
        중앙 디스패처(권장): 여기서 파라미터 호환 → 백업 피벗 → 실제 호출 → 관측 훅 호출을 고정합니다.
        메인 루프에서 handler 직접 부르지 말고 이 함수를 통해 호출하세요.
        """
        # 1) 파라미터 호환(핸들러 시그니처 불일치 방지)
        action_input = self._adapt_action_params(self, action, action_input)

        # 2) 백업 루프 피벗(동일 파일 백업만 반복 시 쓰기 계열로 전환)
        if hasattr(self, "_pivot_duplicate_backup"):
            try:
                action, action_input = await self._pivot_duplicate_backup(action, action_input)
                action_input = self._adapt_action_params(self, action, action_input)
            except Exception as e:
                print(f"[BackupPivot Warning] {e}")

        # 3) 실제 도구 호출
        handler = self.tool_manager.action_handlers.get(action)
        if not handler:
            raise RuntimeError(f"unknown tool: {action}")
        observation = await handler(**(action_input or {}))

        # 4) 관측 후처리(파일 경로 추적/이어쓰기 훅)
        await self._on_action_observed(self, action, action_input, observation)
        return observation
    async def _post_write_boost(self, tool_name: str, action_input: dict, observation: dict) -> None:
        """
        - 목표가 longform_text면 분량 부족분을 append.
        - 비어있거나 <AUTO...> 플레이스홀더만 있으면 '처음 본문'부터 생성.
        - 도구명 하드코딩/자연어 키워드 매칭 없음.
        """
        try:
            cls = getattr(self, "_goal_classification", {}) or {}
            if cls.get("content_kind") != "longform_text":
                return

            # 어디에 썼는지 추적
            path = self._extract_written_file_path(tool_name, action_input or {}, observation or {}) \
                or getattr(self, "last_written_file", "")
            if not path:
                return

            try:
                cur = Path(path).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                cur = ""

            # 분량 힌트
            try:
                from metatools.tool_selector import estimate_length_hint
                min_chars, target_chars = estimate_length_hint(self.user_goal or "", "longform_text")
            except Exception:
                min_chars, target_chars = 1500, 2500

            # 초기 본문 여부
            cur_is_emptyish = (not cur.strip()) or self._is_auto_placeholder(cur)

            # 메시지 구성(LLM 시그니처 변화 안전 호출)

            _msgs = self._build_user_message
            if cur_is_emptyish:
                prompt = (
                    "요청에 부합하는 한국어 장문 본문을 직접 작성하세요.\n"
                    "- 메타 설명/사과/안내문 금지, 본문만.\n"
                    "- 중복 없이 자연스럽게 서사/설명/묘사를 포함.\n"
                    "- 길이 목표: 최소 {min_c}자, 가급적 {tgt_c}자 부근.\n"
                ).format(min_c=min_chars, tgt_c=target_chars)
                resp, _ = await self._call_llm_safe(_msgs(prompt), available_models=getattr(self, "available_models", None), use_tools=False)
                add = (getattr(resp, "text", None) or "").strip()
                if not add: return
                new_text = add
            else:
                prompt = (
                    "다음 한국어 본문을 자연스럽게 '이어지는 다음 부분'으로 확장하세요.\n"
                    "- 이미 있는 문장/문단을 반복하거나 요약하지 말 것.\n"
                    "- 새 사건/세부/감정 묘사로 분량을 확장.\n"
                    "- 메타 코멘트 없이 실제 본문만 출력.\n\n"
                    "[이전 본문 (말미 4000자)]\n" + cur[-4000:]
                )
                resp, _ = await self._call_llm_safe(_msgs(prompt), available_models=getattr(self, "available_models", None), use_tools=False)
                add = (getattr(resp, "text", None) or "").strip()
                if not add: return
                add = self._dedupe_paragraphs(cur, add)
                if not add: return
                new_text = ("\n\n".join([cur.strip(), add]) if cur.strip() else add)

            combined = (cur.rstrip() + "\n\n" + new_text) if cur.strip() else new_text

            if "append_file" in self.tool_manager.action_handlers and cur.strip():
                await self.tool_manager.action_handlers["append_file"](file_path=path, content="\n\n" + new_text)
            elif "write_file" in self.tool_manager.action_handlers:
                # ⚠️ 절대 새 텍스트만 쓰지 말고 항상 병합해서 write
                await self.tool_manager.action_handlers["write_file"](file_path=path, content=combined)
            # self._written_paths 업데이트도 누락 없이
            try:
                norm = str(_P(path).resolve())
                self._written_paths.add(norm)
            except Exception:
                pass

            # append/write 안전 처리(핸들러 존재시 우선 활용)
            if "append_file" in self.tool_manager.action_handlers and not cur_is_emptyish:
                await self.tool_manager.action_handlers["append_file"](file_path=path, content="\n\n" + (new_text[len(cur):] if new_text.startswith(cur) else new_text))
            elif "write_file" in self.tool_manager.action_handlers:
                await self.tool_manager.action_handlers["write_file"](file_path=path, content=new_text)
            else:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new_text)
            try:
                size = len(_P(path).read_text(encoding="utf-8", errors="ignore"))
                print(f"[PostWrite] file='{path}' size={size}")
            except Exception as _e:
                print(f"[PostWrite] size check failed: {str(_e)}")

        except Exception as e:
            print(f"[PostWrite] continuation error: {e}")
    from pathlib import Path
    import time
    def _norm_path(self, p):
        try:
            return str(Path(p).resolve())
        except Exception:
            return p or ""
    def _extract_primary_aux(self, action: str, observation: dict):
        """관찰 결과만으로 주/보조 경로를 추출(도구명 상관없음)."""
        det = observation.get("detail", {})
        if not isinstance(det, dict): det = {}

        path  = observation.get("path") or det.get("path")
        src   = observation.get("src") or det.get("src")
        bpath = observation.get("backup_path") or det.get("backup_path")

        # 백업이면 원본이 primary, 백업본이 aux
        if src and bpath:
            primary = src
            aux = bpath
        else:
            primary = path or src
            aux = bpath

        return self._norm_path(primary), self._norm_path(aux)
    async def _on_action_observed(self, action: str, action_input: dict, observation: dict):
        """액션 수행 직후 1회: 상태 갱신 + 이어쓰기 트리거 + 백업 스로틀."""
        try:
            # 마지막으로 수정된 파일 추적
            if self._looks_write_success(observation):
                det = observation.get("detail", {}) if isinstance(observation.get("detail"), dict) else {}
                self.last_written_file = (det.get("path") or observation.get("path") or "")

            # 백업 스로틀(60초 내 반복)
            if not hasattr(self, "_recent_backup_ts"):
                self._recent_backup_ts = {}
            if self._looks_backup_success(observation):
                primary = observation.get("src") or (det.get("src") if isinstance(det, dict) else None)
                if primary:
                    import time as _t
                    now = _t.time()
                    prev = self._recent_backup_ts.get(primary)
                    self._recent_backup_ts[primary] = now
                    if prev and (now - prev) < 60:
                        print(f"[BackupThrottle] skip repetitive backup hint for {primary}")

            # 파일 작성 성공시에만 이어쓰기 후크
            if self._looks_write_success(observation):
                await self._post_write_boost(action, action_input or {}, observation or {})
        except Exception as e:
            print(f"[Post-Observe] error: {e}")
    async def _post_write_boost(self, tool_name: str, action_input: dict, observation: dict) -> None:
        """
        파일 작성 직후 보강:
        - 목표가 장문이면 분량 부족분을 append.
        - 초기 본문이 짧으면 '처음 본문'부터 생성(이어쓰기 아님).
        - 자연어 키워드 매칭 없이 길이/구조만 사용.
        """
        try:
            from pathlib import Path as _P
            cls = getattr(self, "_goal_classification", {}) or {}
            if cls.get("content_kind") != "longform_text":
                return

            det = observation.get("detail", {}) if isinstance(observation.get("detail"), dict) else {}
            path = det.get("path") or observation.get("path") or getattr(self, "last_written_file", "")
            if not path:
                return

            try:
                cur = _P(path).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                cur = ""

            # 분량 힌트
            try:
                from metatools.tool_selector import estimate_length_hint
                min_chars, max_chars = estimate_length_hint(self.user_goal or "", "longform_text")
            except Exception:
                # 안전 기본값(예: 1500자 ~ 2500자)
                min_chars, max_chars = 1500, 2500

            # --- 초기 본문이 비거나 너무 짧으면 '처음 본문' 생성 ---
            cur_len = len(cur.strip())
            if cur_len < max(50, min_chars // 10):
                from google.genai import types
                seed = (
                    "요청에 맞는 한국어 장문 본문을 작성하세요.\n"
                    "- 메타 코멘트 없이 본문만 출력\n"
                    "- 도입-전개-클라이맥스 흐름으로 자연스럽게 시작\n"
                )
                goal = (self.user_goal or "").strip()
                if goal:
                    seed += "\n[요청]\n" + goal
                resp, _ = await self._call_llm_safe(
                    [types.Content(role="user", parts=[types.Part(text=seed)])],
                    available_models=getattr(self, "available_models", None),
                    use_tools=False,
                )
                add = (getattr(resp, "text", None) or "").strip()
                if add:
                    if "append_file" in self.tool_manager.action_handlers:
                        await self.tool_manager.action_handlers["append_file"](file_path=path, content=("\n\n" if cur else "") + add)
                    elif "write_file" in self.tool_manager.action_handlers:
                        await self.tool_manager.action_handlers["write_file"](file_path=path, content=(cur + ("\n\n" if cur else "") + add))
                    # 갱신
                    try:
                        cur = _P(path).read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        return

            # --- 분량 부족시 이어쓰기(중복 단락 제거 포함) ---
            rounds = 0
            while len(cur) < min_chars and rounds < 2:
                from google.genai import types
                prompt = (
                    "아래 본문의 '다음 부분'을 자연스럽게 이어서 '본문만' 작성하세요.\n"
                    "- 기존 문장/문단 반복 금지\n"
                    "- 사건/디테일/감정 묘사로 전개 확장\n\n"
                    "[이전 본문 (말미 4000자)]\n" + cur[-4000:]
                )
                resp, _ = await self._call_llm_safe(
                    [types.Content(role="user", parts=[types.Part(text=prompt)])],
                    available_models=getattr(self, "available_models", None),
                    use_tools=False,
                )
                add = (getattr(resp, "text", None) or "").strip()
                if not add:
                    break
                add = self._dedupe_paragraphs(cur, add) if hasattr(self, "_dedupe_paragraphs") else add
                if not add:
                    break

                if "append_file" in self.tool_manager.action_handlers:
                    await self.tool_manager.action_handlers["append_file"](file_path=path, content="\n\n" + add)
                elif "write_file" in self.tool_manager.action_handlers:
                    await self.tool_manager.action_handlers["write_file"](file_path=path, content=(cur + "\n\n" + add))
                else:
                    break

                try:
                    cur = Path(path).read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    break
                rounds += 1
        except Exception as _e:
            print(f"[PostWrite] continuation error: {str(_e)}")
    def _safe_append_text(self, path: str, text: str) -> None:
        from pathlib import Path as _P
        _P(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            if text and not text.startswith("\n"):
                f.write("\n\n")
            f.write(text)
    async def _gen_initial_longform(self) -> str:
        from metatools.tool_selector import estimate_length_hint
        min_chars, target_chars = estimate_length_hint(self.user_goal or "", "longform_text")
        try:
            from google.genai import types
            prompt = (
                "아래 사용자 목표를 충실히 반영한 한국어 장문 본문을 바로 출력하세요.\n"
                "- 메타 코멘트/지시어/사과문 금지, 본문만 출력\n"
                f"- 최소 {min_chars}자 이상, 가능하면 {max(min_chars+800, target_chars)}자 내외\n\n"
                "[사용자 목표]\n" + (self.user_goal or "")
            )
            messages = [types.Content(role="user", parts=[types.Part(text=prompt)])]
        except Exception:
            messages = [{"role": "user", "parts": [{"text": self.user_goal or ""}]}]
        resp, _ = await self._call_llm_safe(messages, available_models=getattr(self, "available_models", None), use_tools=False)
        return (getattr(resp, "text", None) or "").strip()
    async def _post_write_boost_effect(self, effect: _Effect) -> None:
        try:
            cls = getattr(self, "_goal_classification", {}) or {}
            if cls.get("content_kind") != "longform_text":
                return
            path = effect.primary_path
            if not path:
                return

            from pathlib import Path as _P
            try:
                cur = _P(path).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                cur = ""

            # 비었거나 플레이스홀더만 있으면: 처음부터 채움 (메타 코멘트 방지)
            if (not cur.strip()) or self._is_auto_placeholder(cur):
                seed = await self._gen_initial_longform()
                if not seed:
                    return
                # overwrite (append_file 없어도 안전)
                if "write_file" in self.tool_manager.action_handlers:
                    await self.tool_manager.action_handlers["write_file"](file_path=path, content=seed.strip())
                else:
                    _P(path).write_text(seed.strip(), encoding="utf-8")
                return

            # 내용이 있지만 분량 부족하면 이어쓰기
            from metatools.tool_selector import estimate_length_hint
            min_chars, _ = estimate_length_hint(self.user_goal or "", "longform_text")
            rounds = 0
            while len(cur) < min_chars and rounds < 2:
                try:
                    from google.genai import types
                    prompt = (
                        "아래 한국어 장문 텍스트의 '다음 부분'을 자연스럽게 이어서 작성하세요.\n"
                        "- 기존 문장/문단 반복 금지, 메타 코멘트 금지\n"
                        "- 사건/감정/디테일 묘사 추가로 분량 확장\n\n"
                        "[이전 본문 (최근 맥락)]\n" + cur[-4000:]
                    )
                    resp, _ = await self._call_llm_safe(
                        [types.Content(role="user", parts=[types.Part(text=prompt)])],
                        available_models=self.available_models,
                        use_tools=False
                    )
                    add = (getattr(resp, "text", None) or "").strip()
                except Exception:
                    add = ""
                if not add:
                    break
                if hasattr(self, "_dedupe_paragraphs"):
                    add = self._dedupe_paragraphs(cur, add) or add
                if not add.strip():
                    break

                # append 우선, 없으면 안전 재쓰기
                if "append_file" in self.tool_manager.action_handlers:
                    await self.tool_manager.action_handlers["append_file"](file_path=path, content="\n\n" + add)
                elif "write_file" in self.tool_manager.action_handlers:
                    await self.tool_manager.action_handlers["write_file"](file_path=path, content=cur + "\n\n" + add)
                else:
                    self._safe_append_text(path, add)

                try:
                    cur = _P(path).read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    break
                rounds += 1
        except Exception as _e:
            print(f"[PostWrite] continuation error: {str(_e)}")
    async def _pivot_duplicate_backup(self, action: str, action_input: dict) -> tuple[str, dict]:
        # 백업이 아니면 그대로
        if action != "backup_file":
            return action, action_input
        from pathlib import Path as _P
        tgt = (action_input or {}).get("file_path") or (action_input or {}).get("src")
        if not tgt:
            return action, action_input
        norm = str(_P(tgt).resolve())

        # 이미 백업했고 아직 '쓰기'가 없었다면 → 곧장 쓰기 계열로 전환
        if norm in getattr(self, "_backed_up_paths", set()) and norm not in getattr(self, "_written_paths", set()):
            # 우선 append 지원 시 append, 아니면 write
            if "append_file" in self.tool_manager.action_handlers:
                return "append_file", {"file_path": norm, "content": "<AUTO_seed>"}
            if "write_file" in self.tool_manager.action_handlers:
                return "write_file", {"file_path": norm, "content": "<AUTO_seed>"}
        return action, action_input
    import os, re
    async def _post_write_boost_effect(self, effect: _Effect) -> None:
        try:
            cls = getattr(self, "_goal_classification", {}) or {}
            if cls.get("content_kind") != "longform_text":
                return
            path = effect.primary_path
            if not path:
                return

            try:
                cur = _P(path).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                cur = ""

            # 플레이스홀더만 들어간 경우 비어있는 본문으로 간주
            if self._is_auto_placeholder(cur):
                cur = ""

            from metatools.tool_selector import estimate_length_hint
            min_chars, _ = estimate_length_hint(self.user_goal or "", "longform_text")

            rounds = 0
            while len(cur) < min_chars and rounds < 2:
                try:
                    from google.genai import types
                    prompt = (
                        "이전 본문의 다음 부분을 자연스럽게 이어서 작성하세요.\n"
                        "중복 없이 새로운 내용을 추가하세요.\n"
                        "파일에 바로 이어붙일 본문만 출력하세요.\n\n"
                        "[이전 본문 (최근 맥락)]\n" + cur[-4000:]
                    )
                    resp, _ = await self._call_llm_safe(
                        [types.Content(role="user", parts=[types.Part(text=prompt)])],
                        available_models=self.available_models,
                        use_tools=False
                    )
                    add = (getattr(resp, "text", None) or "").strip()
                except Exception:
                    add = ""

                if not add:
                    break
                if hasattr(self, "_dedupe_paragraphs"):
                    add = self._dedupe_paragraphs(cur, add) or add
                if not add.strip():
                    break

                self._safe_append_text(path, add)

                try:
                    cur = _P(path).read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    break
                rounds += 1
        except Exception as _e:
            print(f"[PostWrite] continuation error: {str(_e)}")
    async def _pivot_duplicate_backup(self, action: str, action_input: dict) -> tuple[str, dict]:
        # longform이 아니면 패스
        if not self._is_longform_goal():
            return action, action_input
        # 이번 백업 대상 경로 확인
        from pathlib import Path as _P
        tgt = (action_input or {}).get("file_path") or (action_input or {}).get("src")
        if not tgt:
            return action, action_input
        norm = str(_P(tgt).resolve())

        # 이미 백업했고, 그 이후 아직 '쓰기'가 발생하지 않았다면 → write-like로 피벗
        if norm in getattr(self, "_backed_up_paths", set()) and norm not in getattr(self, "_written_paths", set()):
            # 우선순위: append_file > write_file > 그 외 write-like
            preferred = None
            for cand in ("append_file", "write_file"):
                if cand in self.tool_manager.action_handlers:
                    preferred = cand; break
            if not preferred:
                # 카탈로그에서 write-like 탐색(도구명 패턴)
                for d in getattr(self.tool_manager, "all_definitions", []):
                    nm = getattr(d, "name", "")
            if not preferred:
                return action, action_input  # 대체 불가 시 원래대로

            # 파라미터 제안 (스키마 맞춤), 파일 경로는 보존
            from metatools.tool_selector import propose_parameters
            params = await propose_parameters(
                self.tool_manager,
                user_goal=self.user_goal or "",
                tool_name=preferred,
                available_models=getattr(self, "available_models", None),
                classification=getattr(self, "_goal_classification", None),
            ) or {}
            # 스키마에 file_path가 있다면 누락 시 채워줌
            try:
                schema = self.tool_definitions.get(preferred, {})
                allowed = schema.get("allowed", set())
                if "file_path" in allowed and "file_path" not in params:
                    params["file_path"] = tgt
            except Exception:
                pass

            print(f"[BackupThrottle] duplicate backup detected → pivot to '{preferred}'")
            return preferred, params

        return action, action_input
    async def _post_write_boost(self, tool_name: str, action_input: dict, observation: dict) -> None:
        """
        파일에 무엇인가 써진 직후, 목표가 장문이면 부족분을 append로 보강.
        - 도구명/키워드에 의존하지 않고, 관측치/인자에서 '어느 파일'인지만 추출.
        - 현재 파일이 사실상 비었으면(플레이스홀더 포함) '처음부터' 본문을 생성해 채움.
        """
        try:
            cls = getattr(self, "_goal_classification", {}) or {}
            if cls.get("content_kind") != "longform_text":
                return

            # 경로 일반화 추출 (도구명 하드코딩 없이)
            path = self._extract_written_file_path(tool_name, action_input or {}, observation or {}) \
                or getattr(self, "last_written_file", "")
            if not path:
                return

            from pathlib import Path as _P
            try:
                cur = _P(path).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                cur = ""

            # 현재가 '사실상 비어있음' 판단: 공백/플레이스홀더만
            def _only_placeholder_or_empty(s: str) -> bool:
                if not s.strip():
                    return True
                ss = s.strip()
                return ss.startswith("<AUTO") and ss.endswith(">")

            from metatools.tool_selector import estimate_length_hint
            min_chars, _ = estimate_length_hint(self.user_goal or "", "longform_text")

            # 3-1) 비었으면 처음부터 채움
            if _only_placeholder_or_empty(cur):
                seed = await self._gen_initial_longform()
                if not seed:
                    return
                new_text = seed.strip()
                if "append_file" in self.tool_manager.action_handlers and cur.strip():
                    await self.tool_manager.action_handlers["append_file"](file_path=path, content="\n\n" + new_text)
                elif "write_file" in self.tool_manager.action_handlers:
                    await self.tool_manager.action_handlers["write_file"](file_path=path, content=new_text)
                return

            # 3-2) 내용이 있으나 분량이 부족하면 이어쓰기
            rounds = 0
            while len(cur) < min_chars and rounds < 2:
                try:
                    from google.genai import types
                    prompt = (
                        "아래 한국어 장문 텍스트의 '다음 부분'을 자연스럽게 이어서 작성하세요.\n"
                        "- 기존 문장/문단 반복 금지, 메타 코멘트 금지\n"
                        "- 사건/감정/디테일 묘사 추가로 분량 확장\n\n"
                        "[이전 본문 (최근 맥락)]\n" + cur[-4000:]
                    )
                    messages = [types.Content(role="user", parts=[types.Part(text=prompt)])]
                except Exception:
                    messages = [{"role": "user", "parts": [{"text": cur[-4000:]}]}]

                resp, _ = await self._call_llm_safe(messages, available_models=getattr(self, "available_models", None), use_tools=False)
                add = (getattr(resp, "text", None) or "").strip()
                if not add:
                    break

                # 중복 문단 제거 후 append
                add = self._dedupe_paragraphs(cur, add)
                if not add:
                    break

                if "append_file" in self.tool_manager.action_handlers:
                    await self.tool_manager.action_handlers["append_file"](file_path=path, content="\n\n" + add)
                elif "write_file" in self.tool_manager.action_handlers:
                    await self.tool_manager.action_handlers["write_file"](file_path=path, content=cur + "\n\n" + add)
                else:
                    break

                try:
                    cur = _P(path).read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    break
                rounds += 1
        except Exception as _e:
            print(f"[PostWrite] continuation error: {str(_e)}")
    from typing import Optional, Tuple, Dict, Any
    def _auto_placeholder_keys(self, payload: dict) -> list:
        keys = []
        def _is_auto(x):
            return isinstance(x, str) and x.strip().startswith("<AUTO")
        for k, v in (payload or {}).items():
            if _is_auto(v):
                keys.append(k)
        return keys
    async def _decide_brevity_via_llm(self) -> bool:
        """
        사용자가 '짧게/간단히' 등 명시적으로 간결 응답을 요청했는지 LLM에게 의미로 판단시킨다.
        반환: True(간결 요청) | False(간결 요청 아님)
        """
        user_goal = self.user_goal or ""
        instruction = (
            "Return JSON only: {\"brief\": true|false}.\n"
            "Output true ONLY IF the user explicitly asks for brevity (very short/one sentence/brief). "
            "If not explicit, return false."
        )
        prompt = f"{instruction}\n\nUSER_REQUEST:\n{user_goal}"
        try:
            text, _ = await self._call_llm_safe(self._build_user_message(prompt), use_tools=False)
            obj = {}
            try:
                # text는 resp 객체일 수도 있고 str일 수도 있음
                maybe = getattr(text, "text", None)
                obj = json.loads((maybe if isinstance(maybe, str) else (text if isinstance(text, str) else "")).strip())
            except Exception:
                obj = {}
            return bool(obj.get("brief", False))
        except Exception:
            return False
    def _pick_payload_field_fast(self, tool_spec: dict, action_input: dict) -> str | None:
        params = (tool_spec or {}).get("parameters", {}) or {}
        props  = params.get("properties") or {}
        if not isinstance(props, dict): return None
        keys = [k for k,v in props.items() if isinstance(v, dict) and v.get("type") == "string"]
        if not keys: return None

        def looks_path_or_url(s: str) -> bool:
            if not isinstance(s, str): return False
            s = s.strip()
            if s.startswith("http://") or s.startswith("https://"): return True
            # 전체가 경로/확장자에 가까우면 제외
            import re
            if re.search(r"[\\/]", s) and re.search(r"\.[A-Za-z0-9]{1,5}($|\?)", s): return True
            return False

        # 1) 현재 값 중 '<' 포함(HTML/코드) 우선
        htmlish = [k for k in keys if isinstance(action_input.get(k), str) and "<" in action_input.get(k)]
        if htmlish:
            htmlish = [k for k in htmlish if not looks_path_or_url(action_input.get(k))]
            if htmlish:
                return htmlish[0]

        # 2) 경로/URL 제외 후 가장 긴 문자열
        candidates = []
        for k in keys:
            val = action_input.get(k, "")
            if looks_path_or_url(val): continue
            L = len(val) if isinstance(val, str) else -1
            candidates.append((L, k))
        if not candidates:
            return keys[0]
        candidates.sort(reverse=True)
        return candidates[0][1]
    async def _compose_or_enrich(self, user_goal: str, current_text: str, *, min_chars: int) -> str:
        if not isinstance(current_text, str) or len(current_text.strip()) < 200:
            # from-scratch
            prompt = (
                "사용자 요청을 바탕으로 완성 문서를 작성하세요.\n"
                "- 형식을 적절히 선택(요청이 HTML/코드/문서 뉘앙스면 그 형식으로)\n"
                "- 메타 멘트 금지, 결과물만\n"
                f"- 분량: 최소 {min_chars}자\n\n[사용자 요청]\n{user_goal}\n"
            )
        else:
            # preserve-format enrich
            prompt = (
                "CURRENT_TEXT를 형식을 유지하며 대폭 확장하세요.\n"
                "- HTML이면 유효한 HTML 유지, 코드면 코드만\n"
                "- 메타 멘트 금지\n"
                f"- 분량: 최소 {min_chars}자\n\n[USER_GOAL]\n{user_goal}\n\n[CURRENT_TEXT]\n{current_text}\n"
            )
        resp, _ = await self._call_llm_safe(self._build_user_message(prompt), use_tools=False, preferred_model="gemini-2.5-flash")
        text = getattr(resp, "text", None) if not isinstance(resp, str) else resp
        return (text or "").strip()
    async def _enrich_payload_via_llm(self, *, user_goal: str, current_text: str, min_chars: int = 800) -> str:
        """
        형식 불문(HTML/Markdown/코드/순수문서)으로 '현재 텍스트'를 더 길고 충실하게 확장한다.
        - 현재 텍스트의 형식을 유지(HTML이면 HTML, 코드면 코드, MD면 MD, 평문이면 장문 글)
        - 메타 멘트 금지, 결과물만 출력
        """
        # LLM에게 '형식 유지 + 확장'을 강하게 지시
        instruction = (
            "Expand and enrich the CURRENT_TEXT while preserving its exact format/type.\n"
            "- If it's HTML, keep valid, complete HTML (doctype optional if already present).\n"
            "- If it's code, output only the code (same language), no extra commentary.\n"
            "- If it's Markdown, keep Markdown.\n"
            "- If it's plain text, produce a long-form article in the same language.\n"
            "- Do not add explanations or meta text; output only the enriched content.\n"
            f"- Target length: at least {min_chars} characters.\n"
        )
        prompt = (
            f"{instruction}\n\n"
            f"[USER_GOAL]\n{user_goal}\n\n"
            f"[CURRENT_TEXT]\n{current_text or ''}\n"
        )
        resp, _ = await self._call_llm_safe(self._build_user_message(prompt), use_tools=False)
        text = ""
        try:
            text = getattr(resp, "text", None) or ""
        except Exception:
            pass
        if not text and isinstance(resp, str):
            text = resp
        return (text or "").strip()

    async def _maybe_enrich_short_text_payload(self, action: str, action_input: dict, *, min_chars: int = 800, min_threshold: int = 200) -> dict:
        """
        도구 실행 직전: 텍스트 페이로드가 너무 짧으면 자동 보강.
        - 간결 요청이 명시된 경우는 스킵
        - 도구 스키마에서 string 파라미터를 찾아 그 중 하나를 '주요 페이로드'로 선택
        - 현재 길이가 min_threshold 미만이면 LLM으로 보강하여 치환
        """
        try:
            # 간결 요청이면 건드리지 않음
            if await self._decide_brevity_via_llm():
                return action_input

            tool_spec = self.tool_manager.tool_catalog.get(action, {}) if hasattr(self.tool_manager, "tool_catalog") else {}
            key = self._pick_payload_field_fast(tool_spec, action_input or {})
            if not key:
                return action_input

            cur = action_input.get(key, "")
            if not isinstance(cur, str):
                return action_input

            if len(cur.strip()) >= min_threshold:
                return action_input

            enriched = await self._enrich_payload_via_llm(user_goal=self.user_goal or "", current_text=cur, min_chars=min_chars)
            if enriched and len(enriched) > len(cur):
                new_args = dict(action_input or {})
                new_args[key] = enriched
                return new_args
            return action_input
        except Exception:
            return action_input
    async def _call_llm_safe(self, messages, **kwargs):
        """
        - call_gemini_agent()에 안전 래핑
        - 429(RESOURCE_EXHAUSTED) 시 RetryInfo.retryDelay 만큼 대기 후 동일 모델 1회 재시도
        그래도 실패하면 available_models를 회전시켜 폴백 시도
        - 동일 이터레이션 내 동일 프롬프트는 캐시로 중복 호출 차단
        """
        import asyncio, random
        from inspect import signature

        # ✅ dedup 활성 여부 (기본 False)
        dedup = bool(kwargs.pop("dedup", False))

        # 이터레이션 캐시 준비
        cache = getattr(self, "_iter_prompt_cache", None)
        if not isinstance(cache, dict):
            self._iter_prompt_cache = {}
            cache = self._iter_prompt_cache

        # 프롬프트 중복 차단: 메시지 해시
        try:
            fp = self._messages_fingerprint(messages)
        except Exception:
            fp = None

        # ✅ dedup이 True일 때만 캐시 확인
        if dedup and fp and fp in cache:
            cached = cache[fp]
            print("[LLM] dedup: return cached response for identical prompt in this iteration")
            return cached["response"], cached["model"]
        # 시그니처 필터링(프로젝트의 call_gemini_agent에 맞춰 kwargs 정리)
        try:
            sig = signature(call_gemini_agent)  # 기존 프로젝트 함수
            filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
        except Exception:
            filtered = dict(kwargs)

        # 모델 리스트 확보(없으면 빈 리스트)
        models = list(filtered.get("available_models") or getattr(self, "available_models", []) or [])

        async def _attempt(call_kwargs):
            # 실제 호출: 시그니처에 맞춘 kwargs만 전달
            return await call_gemini_agent(messages, **call_kwargs)
        # 호출 성공 시: ✅ dedup=True일 때만 캐시 저장
        try:
            if dedup and fp:
                cache[fp] = {"response": response, "model": model}
        except Exception:
            pass
        # 1차 시도
        try:
            response, model = await _attempt(filtered)
            if fp:
                cache[fp] = {"response": response, "model": model}
            return response, model
        except Exception as e:
            msg = str(e)
            is_quota = ("RESOURCE_EXHAUSTED" in msg) or ("429" in msg)

            if is_quota:
                # RetryInfo 대기 + 지터
                delay = self._parse_retry_delay_sec(e) + random.uniform(0.2, 0.8)
                print(f"[LLM] Quota hit. Sleeping ~{delay:.1f}s per RetryInfo...")
                await asyncio.sleep(delay)

                # 같은 모델로 1회 재시도
                try:
                    response, model = await _attempt(filtered)
                    if fp:
                        cache[fp] = {"response": response, "model": model}
                    return response, model
                except Exception:
                    # 다음 모델로 폴백 (available_models 회전)
                    if len(models) >= 2:
                        rotated = models[1:] + models[:1]
                        filtered["available_models"] = rotated
                        try:
                            response, model = await _attempt(filtered)
                            # 우선순위 갱신 훅이 있으면 호출 (없으면 무시)
                            bump = getattr(self, "_bump_model_priority", None)
                            if callable(bump):
                                try:
                                    bump(model)
                                except Exception:
                                    pass
                            if fp:
                                cache[fp] = {"response": response, "model": model}
                            return response, model
                        except Exception:
                            pass
            # 429가 아니거나 모든 재시도 실패 → 원래 예외 전파
            raise
    def _get_current_user_text(self) -> str:
        try:
            if hasattr(self, "last_user_message") and self.last_user_message:
                return self.last_user_message
            if hasattr(self, "user_goal") and self.user_goal:
                return self.user_goal
            return ""
        except Exception:
            return ""
    def _save_state(self):
        """현재 모델 우선순위를 파일에 저장합니다."""
        try:
            with open(AGENT_STATE_PATH, 'w', encoding='utf-8') as f:
                json.dump({"model_priority_list": self.available_models}, f, indent=2)
                print(f"💾 모델 우선순위가 파일에 저장되었습니다: {self.available_models}")
        except Exception as e:
            print(f"⚠️ Agent 상태 저장 실패: {e}")        
    async def _initialize(self):
        """세션 확인, DB 스키마 로딩, 초기 프롬프트 구성 등 초기화 작업을 수행합니다."""
        session_exists = self.db.query(Session).filter(Session.session_id == self.session_id).first()
        if not session_exists:
            default_user = self.db.query(User).filter(User.user_name == '신종혁').first()
            self.current_user_id = default_user.user_id if default_user else 1
            new_session = Session(session_id=self.session_id, owner_id=self.current_user_id)
            self.db.add(new_session); self.db.commit()
        else:
            self.current_user_id = session_exists.owner_id
        
        schema_response = await asyncio.to_thread(get_db_schema)
        if schema_response["status"] != "success":
            raise Exception(f"DB 스키마 로딩 실패: {schema_response.get('message')}")
        schema_for_prompt = json.dumps(schema_response.get("schema", {}), indent=2, ensure_ascii=False)
        
        initial_prompt_text = build_prompt(
            f"**Current DB Schema:**\n```json\n{schema_for_prompt}\n```\n\n"
            f"**Current Session ID:** `{self.session_id}`\n\n"
            f"**사용자 요청 (최종 목표):**\n\"{self.user_goal}\""
        )
        self.prompt_content.append(types.Content(role="user", parts=[types.Part(text=initial_prompt_text)]))
    async def _call_model_and_parse_with_retry(self, max_retries: int = 1):
        """LLM을 호출하고, 파싱 실패 시 보정(최대 3회) 후에도 안 되면 안전 JSON으로 귀결."""
        retry_count = 0
        raw_text = ""

        while retry_count <= max_retries:
            response, successful_model = await call_gemini_agent(
                prompt_content=self.prompt_content,
                available_models=self.available_models,
                tools=self.tool_manager.tools
            )

            if not response:
                # 안전 JSON 반환 (최소한 final_response 실행)
                safe_json = {
                    "Thought": "모델 호출에 실패하여 안전 종료합니다.",
                    "Action": {
                        "tool_name": "final_response",
                        "parameters": {
                            "answer": "잠시 오류가 있었어요. 다시 시도해 주세요. (모델 호출 실패)"
                        }
                    }
                }
                return safe_json["Thought"], safe_json["Action"]["tool_name"], safe_json["Action"]["parameters"], None

            if successful_model and successful_model != self.available_models[0]:
                print(f"[💡 Model Priority Updated] '{successful_model}'을(를) 최우선 모델로 변경합니다.")
                try:
                    self.available_models.remove(successful_model)
                except ValueError:
                    pass
                self.available_models.insert(0, successful_model)
                self._save_state()

            candidate = response.candidates[0]
            parts = getattr(candidate.content, "parts", [])
            if not parts:
                # 안전 JSON 반환
                safe_json = {
                    "Thought": "모델이 빈 응답을 반환하여 안전 종료합니다.",
                    "Action": {
                        "tool_name": "final_response",
                        "parameters": {
                            "answer": "잠시 오류가 있었어요. 다시 시도해 주세요. (빈 응답)"
                        }
                    }
                }
                return safe_json["Thought"], safe_json["Action"]["tool_name"], safe_json["Action"]["parameters"], None

            part = parts[0]
            function_call = getattr(part, "function_call", None)
            if function_call:
                action = getattr(function_call, "name", None)
                action_input = dict(getattr(function_call, "args", {}))
                thought = action_input.get('thought', 'Function Call에서 생각 추출')
                if action:
                    return thought, action, action_input, candidate

            raw_text = getattr(part, "text", "")
            if raw_text:
                # 1차: 직접 파싱
                obj = self._coerce_json_from_text(raw_text)
                if isinstance(obj, dict) and isinstance(obj.get("Action"), dict):
                    thought = obj.get("Thought", "")
                    action = obj["Action"].get("tool_name")
                    action_input = obj["Action"].get("parameters", {}) or {}
                    if action:
                        return thought, action, action_input, candidate

                # 2차: JSON 보정(최대 3회)
                print("⚠️ 모델이 유효 JSON을 내놓지 않았습니다. 보정 루프를 시작합니다...")
                repaired = await self._repair_json_via_llm(raw_text, max_attempts=3)
                if isinstance(repaired, dict):
                    thought = repaired.get("Thought", "")
                    action = (repaired.get("Action") or {}).get("tool_name")
                    action_input = (repaired.get("Action") or {}).get("parameters", {}) or {}
                    if action:
                        return thought, action, action_input, candidate

            # 3차: 바깥 재시도(프롬프트 보강 후 1회 재호출)
            retry_count += 1
            if retry_count > max_retries:
                break

            print(f"⚠️ 모델이 JSON을 반환하지 않아 5초 후 재시도합니다... (시도 {retry_count}/{max_retries})")
            await asyncio.sleep(5)

            def _as_content(obj):
                if isinstance(obj, types.Content):
                    return obj
                if isinstance(obj, str):
                    return types.Content(role="model", parts=[types.Part(text=obj)])
                txt = getattr(obj, "text", None)
                if isinstance(txt, str):
                    return types.Content(role="model", parts=[types.Part(text=txt)])
                content = getattr(obj, "content", None)
                if content and getattr(content, "parts", None):
                    return content
                return types.Content(role="model", parts=[types.Part(text=str(obj))])


            self.prompt_content.append(_as_content(candidate))
            self.prompt_content.append(
                types.Content(role="user", parts=[types.Part(text="반드시 단일 JSON 객체로만 응답하세요. {\"Thought\":\"...\",\"Action\":{...}}")])
            )

        # 최종 실패: 안전 JSON 폴백
        safe_json = {
            "Thought": "모델 JSON 생성이 연속으로 실패했으므로 안전 종료합니다.",
            "Action": {
                "tool_name": "final_response",
                "parameters": {
                    "answer": "잠시 오류가 있었어요. 한 번만 다시 시도해 주세요. (JSON 복구 루프 실패)"
                }
            }
        }
        return safe_json["Thought"], safe_json["Action"]["tool_name"], safe_json["Action"]["parameters"], None
    async def _propose_first_step(self):
        """
        [최적화] 사용자 목표를 분석하여 (1)의도 분류와 (2)도구 계획을 단일 LLM 호출로 처리합니다.
        간단한 대화는 도구 사용 없이 바로 다음 단계로 넘어갑니다.
        """
        user_goal_text = (self.user_goal or "").strip()
        print(f"[🚀 Proposing First Step] 통합 분석 및 계획을 시작합니다: '{user_goal_text}'")

        try:
            # --- 1. 단일 분석/계획을 위한 프롬프트 구성 ---
            _all_defs = [d for d in getattr(self.tool_manager, "all_definitions", []) if getattr(d, "name", "")]
            _BANNABLE = {"final_response", "ask_followup_question", "clarify", "reflect"}
            BANNED_TERMINALS = {d.name for d in _all_defs if d.name in _BANNABLE}
            _defs_no_terminals = [d for d in _all_defs if d.name not in BANNED_TERMINALS]

            def _dump_catalog(defs):
                # (이 헬퍼 함수는 기존과 동일하게 유지)
                meta = getattr(self.tool_manager, "tool_meta", {})
                parts = []
                for d in defs:
                    name = getattr(d, "name", "")
                    desc = getattr(d, "description", "")
                    m = (meta.get(name, {}) or {})
                    tags = m.get("tags", [])
                    dep = m.get("dependency_graph", {}) or {}
                    nodes = dep.get("nodes") or []
                    covers = [n.get("tool_name") for n in nodes if isinstance(n, dict) and n.get("tool_name")]
                    lines = [f"- {name} : {desc}", f"  tags: {tags}"]
                    if covers:
                        lines.append(f"  covers: {', '.join(covers)}")
                    parts.append("\n".join(lines))
                return "\n".join(parts)

            tool_catalog_dump = _dump_catalog(_defs_no_terminals)

            # 통합 프롬프트
            unified_prompt = (
                "당신은 사용자의 요청을 분석하고 실행 계획을 세우는 '지능형 플래너'입니다.\n"
                "요청을 분석하여 (1)의도 분류와 (2)실행 계획을 하나의 JSON 객체로 응답하세요.\n\n"
                "[의도 분류 기준]\n"
                "- 'chat': 간단한 인사, 감정 표현, 짧은 질문 등 도구가 필요 없는 일반 대화.\n"
                "- 'task': 파일 작업, 정보 조회, 코드 생성 등 도구 사용이 필요한 명확한 작업.\n\n"
                "[사용 가능한 도구 목록]\n"
                f"{tool_catalog_dump}\n\n"
                "[규칙]\n"
                "- 의도가 'chat'이면, 'steps'는 빈 배열 `[]`이어야 합니다.\n"
                "- 의도가 'task'이면, 목표 달성에 필요한 도구 목록을 'steps'에 순서대로 나열하세요.\n"
                "- 가능한 경우, 여러 단계를 포괄하는 매크로 도구를 우선적으로 사용하세요.\n"
                "- 종료/질문 관련 도구(final_response 등)는 'steps'에 포함하지 마세요.\n"
                "- 출력은 반드시 아래 형식의 순수 JSON 객체여야 합니다.\n\n"
                "[사용자 목표]\n"
                f'"{user_goal_text}"\n\n'
                "[출력 형식 - JSON Only]\n"
                "{\n"
                '  "classification": {\n'
                '    "intent": "chat" | "task"\n'
                '  },\n'
                '  "thought": "계획에 대한 간단한 생각.",\n'
                '  "steps": ["도구명1", "도구명2", ...]\n'
                "}"
            )

            # --- 2. 단일 LLM 호출 ---
            response, _ = await call_gemini_agent(
                [types.Content(role="user", parts=[types.Part(text=unified_prompt)])],
                available_models=self.available_models,
                use_tools=False
            )

            if (not response) or (not getattr(response, "text", None)):
                print("[🔍 Plan] 1차 실패 → 보수 설정으로 재시도합니다.")
                response, successful_model = await call_gemini_agent(
                    [types.Content(role="user", parts=[types.Part(text=unified_prompt)])],
                    available_models=self.available_models,
                    use_tools=False,
                    temperature=0  # llm_utils.call_gemini_agent 시그니처에 존재
                )
                if (not response) or (not getattr(response, "text", None)):
                    print("[🔍 Plan] 통합 분석/계획 생성 실패. 일반 모드로 진행합니다.")
                    # (선택) 이후 로직 일관성을 위해 빈 플랜 캐시
                    self._latest_plan_obj = {"steps": []}
                    return None, None

            # --- 3. 통합 응답 파싱 및 처리 ---
            raw_text = getattr(response, "text", "")
            plan_obj = self._coerce_json_from_text(raw_text)

            if not isinstance(plan_obj, dict):
                print(f"[🔍 Plan] 유효한 JSON 응답 파싱 실패. Raw: {raw_text}")
                self._latest_plan_obj = {"steps": []}
                return None, None

            # 3-1. 분류 결과 저장 및 확인
            classification = plan_obj.get("classification", {})
            self._goal_classification = classification # 에이전트 상태에 저장
            intent = classification.get("intent")
            
            print(f"[Classifier] 모델이 의도를 '{intent}'로 판단했습니다.")

            # 3-2. 도구 실행 계획 추출
            steps = plan_obj.get("steps")
            if not isinstance(steps, list):
                print("[🔍 Plan] 'steps'가 유효한 리스트가 아닙니다. 계획을 비웁니다.")
                steps = []

            # 3-3. 간단한 대화(chat)로 분류되면, 계획 없이 바로 종료
            if intent == 'chat' or not steps:
                print("💬 간단한 대화로 판단하여, 도구 계획 없이 일반 응답을 진행합니다.")
                # None을 반환하여 메인 루프가 일반 LLM 호출을 하도록 유도
                return None, None
            
            # --- 4. (기존 로직과 유사) 계획이 있을 경우 첫 단계 제안 ---
            # 카탈로그에 존재하는 유효한 도구만 필터링
            catalog_set = {d.name for d in _all_defs}
            remaining_steps = [s for s in steps if isinstance(s, str) and s in catalog_set]

            if not remaining_steps:
                print("[🔍 Plan] 계획된 단계가 있지만, 유효한 도구가 없습니다.")
                return None, None

            # (선택적) 기존의 set_cover 로직을 여기에 적용하여 계획을 더 최적화할 수 있습니다.
            # remaining_steps = _apply_set_cover(remaining_steps)
            # ✅ 여기 추가: 정규화된 계획을 세션 상태로 저장 (이후 _get_planned_steps()에서 사용)
            self._latest_plan_obj = {"steps": list(remaining_steps)}
            # 첫 번째 단계를 제안
            first_tool = remaining_steps[0]
            print(f"[✅ First Step Proposed] AI가 첫 단계로 '{first_tool}' 도구를 제안했습니다.")

            # 파라미터 생성
            params = await propose_parameters(
                self.tool_manager,
                user_goal=user_goal_text,
                tool_name=first_tool,
                available_models=self.available_models,
            )

            return first_tool, params

        except Exception as e:
            import traceback
            print(f"[First Step Fatal] 첫 단계 제안 중 심각한 오류 발생: {e}")
            traceback.print_exc()
            return None, None
    def _emit_thought(self, text: str) -> None:
        """
        Thought UI/디버깅 스트림으로 보냅니다. event_bus가 없으면 로그로 fallback.
        """
        try:
            bus = getattr(self, "event_bus", None)
            if bus:
                bus.emit("agent_thought", {"text": str(text)})
            else:
                print(f"[THOUGHT] {text}")
        except Exception as e:
            print(f"[Thought Emit Warning] {e}")
    def _sanitize_action_input_and_emit_thought(self, action_input: dict, top_level_json: dict | None = None) -> dict:
        """
        - top_level_json의 Thought(또는 thought)를 이벤트로 내보내고
        - 도구 파라미터에서 메타키를 제거합니다.
        """
        META_KEYS = {"Thought", "thought", "notes", "debug", "reason", "rationale"}
        # 1) Thought 이벤트
        try:
            if isinstance(top_level_json, dict):
                t = top_level_json.get("Thought") or top_level_json.get("thought")
                if t:
                    self._emit_thought(t)
        except Exception as e:
            print(f"[Thought Extract Warning] {e}")

        # 2) META 키 제거
        try:
            clean = {}
            for k, v in (action_input or {}).items():
                if k not in META_KEYS:
                    clean[k] = v
            return clean
        except Exception as e:
            print(f"[META Filter Warning] {e}")
            # 문제 발생 시 원본을 그대로 반환
            return action_input or {}
    def _is_longform_goal(self) -> bool:
        """현재 목표가 장문 텍스트 생성인지 확인(분류기 결과 참조)."""
        cls = getattr(self, "_goal_classification", {}) or {}
        return cls.get("content_kind") == "longform_text"
    def _as_content(self, x):
        # 이미 딕셔너리 형태의 content면 그대로 반환 (호환성을 위해 추가)
        if isinstance(x, dict) and "parts" in x and "role" in x:
            return 
        # 문자열인지, .text 속성이 있는지 등을 확인하여 텍스트 추출
        text_content = ""
        if isinstance(x, str):
            text_content = x
        elif hasattr(x, "text") and isinstance(getattr(x, "text"), str):
            text_content = getattr(x, "text")
        else:
            text_content = str(x)
        # 최종적으로 딕셔너리 형태로 만들어 반환
        return {"role": "model", "parts": [text_content]}
    def _calc_target_chars(self, cap_tokens=65536, reserve=12000, tok_to_char=1.6):
        usable = max(2048, int(cap_tokens * 0.6) - reserve)
        return int(usable * tok_to_char)
    async def _execute_tool(self, action: str, params: dict):
        """
        모든 도구 호출은 이 함수로만 들어오게 한다.
        1) (형식 불문) 텍스트 페이로드 자동 보강(표준)
        2) ensure_args 정규화
        3) 실패 시 보강(강) 1회 → 재-ensure
        4) 최종적으로 _call_tool 로 실행 (재귀 금지)
        """
        params = dict(params or {})
        tool_spec = getattr(self.tool_manager, "tool_catalog", {}).get(action, {}) if hasattr(self.tool_manager, "tool_catalog") else {}

        # 1) 표준 보강(간결 요청 아니고, 텍스트가 짧으면)
        if hasattr(self, "_maybe_enrich_text_payload_strong"):
            params = await self._maybe_enrich_text_payload_strong(
                action=action,
                action_input=params,
                tool_spec=tool_spec,
                min_threshold=5000,              # "짧음" 판단 기준(문자)
                min_chars=self._calc_target_chars()  # 출력 상한 고려해 동적으로 계산
            )

        # 2) 정규화
        ok, normed, err = self.tool_manager.ensure_args(action, params)
        if ok:
            return await self._call_tool(action, normed)

        # 3) 실패 시 '강 보강' 1회
        if hasattr(self, "_maybe_enrich_text_payload_strong"):
            params2 = await self._maybe_enrich_text_payload_strong(
                action=action,
                action_input=params,
                tool_spec=tool_spec,
                min_threshold=8000,                              # 더 엄격
                min_chars=max(self._calc_target_chars(), 30000)  # 목표 길이 상향(문자)
            )
            ok2, normed2, err2 = self.tool_manager.ensure_args(action, params2)
            if ok2:
                return await self._call_tool(action, normed2)

        # 4) 최후 수단: 있는 그대로 실행(또는 여기서 실패 반환)
        return await self._call_tool(action, params)



    async def _run_iteration(self, iteration: int, first_step_proposal: tuple = (None, None)):
        """
        한 사이클 실행:
        - 첫 이터레이션에 제안된 단계가 있으면 우선 사용
        - 모델 호출로 action, action_input, thought 결정
        - 도구 실행은 kwargs로만 호출
        - (존재 시) generated_definitions 기반 스키마 검증 수행
        - 완료 후 prompt_content에 function_call/function_response 기록
        - 백그라운드로 optimization materializer 트리거
        - 실행 결과(성공/실패, 지연) 를 ToolRegistry에 report하여 이후 추천 품질 향상
        """
        import json
        print(f"\n======== Agent Iteration {iteration + 1} (Session: {self.session_id}) ========")

        thought, action, action_input, candidate = None, None, None, None
        observation_result = None  # ← 미리 선언해 UnboundLocalError 방지

        # 1) 첫 이터레이션: 제안된 단계 우선 사용
        proposed_action, proposed_input = first_step_proposal
        if iteration == 0 and proposed_action:
            print(f"💡 제안된 첫 단계({proposed_action})를 사용하여 이터레이션을 시작합니다.")
            action = proposed_action
            action_input = proposed_input or {}
            thought = action_input.get("thought", f"제안된 첫 단계 '{action}'을 실행합니다.")
            try:
                candidate_content_part = types.Part(function_call=types.FunctionCall(name=action, args=action_input))
                candidate = types.Candidate(content=types.Content(role="model", parts=[candidate_content_part]))
            except NameError:
                candidate = None
        else:
            # 2) 이후 이터레이션: 모델 호출
            thought, action, action_input, candidate = await self._call_model_and_parse_with_retry()

        if not action:
            error_message = (action_input or {}).get("error", "알 수 없는 파싱 오류")
            # observation_result는 항상 dict로! (크래시 방지)
            return True, thought, "error", {"error": error_message}, {
                "status": "error",
                "reason": error_message
            }
        action_input = self._sanitize_action_input_and_emit_thought(
            action_input,
            {"Thought": thought} if thought else None
        )
        print(f"[✅ Agent Response] Thought: {thought}")
        print(f"  Action: {action}, Input: {action_input}")

        # 3) 워크플로우 기록(도구 호출 전) — thought 제거
        if action != "final_response":
            try:
                params_for_workflow = dict(action_input or {})
                params_for_workflow.pop("thought", None)
                self.session_workflow.append(json5.dumps({"tool_name": action, "parameters": params_for_workflow}))
            except Exception as e:
                print(f"[Workflow Log Warning] {e}")
        # (핸들러 실행 이후로 이동한 PostWrite는 아래에서 호출)
        if isinstance(observation_result, dict) and observation_result.get("status") == "ok":
            await self._post_write_boost(action, action_input or {}, observation_result or {})
            # 길이 충족 시 자동 종료(모델이 final_response를 못 내도 마무리)
            try:
                cls = getattr(self, "_goal_classification", {}) or {}
                if cls.get("content_kind") == "longform_text":
                    obs = observation_result
                    det = obs.get("detail", {}) if isinstance(obs.get("detail"), dict) else {}
                    path = det.get("path") or obs.get("path") or getattr(self, "last_written_file", "")
                    if path:
                        from metatools.tool_selector import estimate_length_hint
                        min_chars, max_chars = estimate_length_hint(self.user_goal or "", "longform_text")
                        try:
                            from pathlib import Path as _P
                            cur = _P(path).read_text(encoding="utf-8", errors="ignore")
                        except Exception:
                            cur = ""
                        if len(cur) >= min_chars:
                            final_answer = f"파일에 결과를 저장했습니다: {path}"
                            return True, thought, action, {"answer": final_answer}, {"answer": final_answer}
            except Exception as _e:
                    print(f"[PostWrite] finalize check skipped: {_e}")
        # 4) 최종 응답 처리
        if action == "final_response":
            final_answer = (action_input or {}).get("answer", "")
            print(f"[🏁 Final Answer] {final_answer}")

                # 모델 기록
            try:
                if candidate:
                    self.prompt_content.append(self._as_content(candidate))
                else:
                    self.prompt_content.append(
                        types.Content(
                            role="model",
                            parts=[types.Part(function_call=types.FunctionCall(name=action, args=action_input or {}))]
                        )
                    )
                self.prompt_content.append(
                    types.Content(
                        role="user",  # function_response는 user 역할 메시지에 담는 게 규약
                        parts=[types.Part(function_response=types.FunctionResponse(
                            name=action,
                            response={"result": {"answer": final_answer}}
                        ))]
                    )
                )
            except NameError:
                pass

            # 백그라운드 최적화 트리거
            try:
                import asyncio
                if not hasattr(self, "optim_manager") or self.optim_manager is None:
                    from optimization_manager import OptimizationManager
                    self.optim_manager = OptimizationManager()
                generate_tool_definition_and_code = globals().get("generate_tool_definition_and_code")
                register_newly_generated_tool = globals().get("register_newly_generated_tool")
                handle_reload_skills = globals().get("handle_reload_skills")
                if generate_tool_definition_and_code and register_newly_generated_tool and handle_reload_skills:
                    workflow_str_list = getattr(self, "session_workflow", [])
                    user_goal_for_optimization = getattr(self, "user_goal", None)
                    asyncio.create_task(
                        self.optim_manager.record_and_maybe_materialize(
                            workflow_str_list,
                            user_goal_for_optimization,
                            generate_tool_definition_and_code=generate_tool_definition_and_code,
                            register_newly_generated_tool=register_newly_generated_tool,
                            handle_reload_skills=handle_reload_skills
                        )
                    )
            except Exception as e:
                print(f"[Optimization Hook Error] {e}")

            return True, thought, action, action_input, {"answer": final_answer}
        # ✅ 짧은 텍스트 자동 보강(형식 불문)
        action_input = await self._maybe_enrich_short_text_payload(action, action_input or {}, min_chars=800, min_threshold=200)

        # 이후 정규화 → 실제 도구 호출
        ok, normed, err = self.tool_manager.ensure_args(action, action_input or {})
        # (도구 실행 직전) 레지스트리 우선 치환 시도
        try:
            action, action_input, _override_meta = self._maybe_override_with_registry_tool(action, action_input)
            if _override_meta.get("overridden"):
                print(f"[Registry Override] {json.dumps(_override_meta, ensure_ascii=False)}")
        except Exception as e:
            print(f"[Registry Override Warning] {e}")
        # 2) 핸들러 실행 '직전'에 넣을 가드: 플레이스홀더로는 절대 실행하지 않기
        # 위치: _run_iteration 내부, 실제 handler 호출 직전에 삽입
        if isinstance(action_input, dict):
            # longform 목표이고, 입력에 <AUTO...> 플레이스홀더가 하나라도 있으면 대체
            cls = getattr(self, "_goal_classification", {}) or {}
            if cls.get("content_kind") == "longform_text":
                keys = self._auto_placeholder_keys(action_input)
                if keys:
                    seed = await self._gen_initial_longform()
                    for k in keys:
                        action_input[k] = seed
        # 🔒 중복 백업 → 쓰기류로 자동 피벗
        try:
            action, action_input = await self._pivot_duplicate_backup(action, action_input)
        except Exception as e:
            print(f"[BackupThrottle Warning] {e}")
        action, action_input, meta = self._maybe_override_with_registry_tool(action, action_input or {})
        if meta.get("overridden"):
            print(f"[Registry Override] {meta}")
        destination = await self._llm_decide_destination()

        # 2) LLM으로 최종 파라미터 작성 (도구 스키마 전달, 경로명 하드코딩/검출 없음)
        tool_spec = self.tool_manager.tool_catalog.get(action, {})
        authored = await self._llm_author_parameters(
            tool_name=action,
            tool_spec=tool_spec,
            destination=destination,
            proposed_params=action_input or {}
        )

        # 3) authored 파라미터로 치환 (이제부터는 이 값만 사용)
        action_input = authored or {}
        # 5) 도구 실행
        if action in self.tool_manager.action_handlers:
            handler = self.tool_manager.action_handlers[action]
            # backup_file : 같은 파일은 세션 내 중복 백업 금지
            try:
                if action == "backup_file":
                    action, action_input = await self._pivot_duplicate_backup(action, action_input)

                # 실행
                observation_result = await self._execute_tool(action, action_input or {})

                # 효과 정규화 및 상태 갱신
                eff = self._normalize_effect(action, observation_result)
                if eff.kind == "file_backup":
                    self._backed_up_paths.add(eff.primary_path)
                elif eff.kind == "file_mutation":
                    self._written_paths.add(eff.primary_path)
                    await self._post_write_boost_effect(eff)  # ★ 여기서만 1회 호출
            except Exception as _e:
                print(f"[Backup Guard Warning] {str(_e)}")
                # backup_file 성공 시 세션 가드에 추가
                try:
                    if action == "backup_file" and isinstance(observation_result, dict) and observation_result.get("status") == "ok":
                        from pathlib import Path as _P
                        _src = observation_result.get("src") or (action_input or {}).get("file_path") or (action_input or {}).get("src")
                        if _src:
                            self._backed_up_paths.add(str(_P(_src).resolve()))
                except Exception as _e:
                    print(f"[Backup Guard Update Warning] {str(_e)}")
            # 레지스트리 핸들 준비(지연/성공 보고용)
            try:
                from tool_registry import ToolRegistry
                if not hasattr(self, "_tool_registry") or self._tool_registry is None:
                    self._tool_registry = ToolRegistry()
                else:
                    self._tool_registry.refresh()
            except Exception:
                pass

            # 5-a) (선택) 스키마 검증
            do_schema_check = hasattr(self, "_validate_action_input") and hasattr(self, "tool_definitions")
            if do_schema_check:
                try:
                    ok, schema_err = self._validate_action_input(action, action_input or {})
                except Exception as e:
                    ok, schema_err = True, {}
                    print(f"[Schema Validate Warning] {e}")
                if not ok:
                    observation_result = schema_err

            if observation_result is None:
                # kwargs로 호출 + 지연 측정
                import time as _time, inspect as _inspect
                start_ts = _time.perf_counter()
                try:
                    sig = _inspect.signature(handler)
                    observation_result = await self._execute_tool(action, action_input or {})
                except TypeError as e:
                    try:
                        sig_str = str(_inspect.signature(handler))
                    except Exception:
                        sig_str = "(signature unavailable)"
                    observation_result = {
                        "status": "error",
                        "reason": f"핸들러 인자 불일치로 실행 실패: {e}",
                        "hint": "Use exactly the parameter names from generated_definitions.json.",
                        "handler_signature": sig_str,
                        "given_params": action_input or {}
                    }
                except Exception as e:
                    observation_result = {"status": "error", "reason": f"핸들러 실행 중 예외: {e}"}

        else:
            observation_result = {"status": "error", "reason": f"알 수 없는 Agent 액션: '{action}'."}

        print(f"[✅ Observation] {json.dumps(observation_result, ensure_ascii=False, indent=2)}")
        await self._on_action_observed(action, action_input or {}, observation_result or {})
        # 도구 실행 → observation_result 출력 직후에 아래를 추가
        try:
            if self._is_longform_goal() and self._extract_written_file_path(action, action_input or {}, observation_result or {}):
                await self._post_write_boost(action, action_input or {}, observation_result or {})
        except Exception as e:
            print(f"[PostWrite] continuation error: {e}")
        effect = self._normalize_effect(observation_result or {})
        self._effects.append(effect)

        try:
            if effect.kind == "file_mutation" and effect.ok:
                await self._post_write_boost_effect(effect)

                # 길이 충족 시 조기 종료
                try:
                    from metatools.tool_selector import estimate_length_hint
                    min_chars, _ = estimate_length_hint(self.user_goal or "", "longform_text")
                    cur = _P(effect.primary_path).read_text(encoding="utf-8", errors="ignore")
                    if len(cur) >= min_chars:
                        ans = f"저장 완료: {effect.primary_path}"
                        return True, thought, action, {"answer": ans}, {"answer": ans}
                except Exception:
                    pass
        except Exception as _e:
            print(f"[PostWrite Hook] {str(_e)}")
        # -- Post-write continuation hook: write가 너무 짧으면 자동 보강(append) --
        try:
            written_path = self._extract_written_file_path(action, action_input or {}, observation_result or {})
            if written_path:
                from pathlib import Path as _P
                self._written_paths.add(str(_P(written_path).resolve()))
                self.last_written_file = written_path or self.last_written_file
        except Exception:
            pass
        try:
            if action == "backup_file" and isinstance(observation_result, dict) \
            and observation_result.get("status") == "ok":
                from pathlib import Path as _P
                _src = observation_result.get("src") or (action_input or {}).get("file_path") \
                    or (action_input or {}).get("src")
                if _src:
                    self._backed_up_paths.add(str(_P(_src).resolve()))
        except Exception as _e:
            print(f"[Backup Guard Update Warning] {str(_e)}")
        # -------------------------------------------------------------------------
        # 6) 부가 상태 갱신
        try:
            if action == "write_file":
                # 우선 관찰 결과(detail.path 등)에서 경로를 읽고, 없으면 입력 인자 중 'path' 포함 키를 탐색
                obs = observation_result if isinstance(observation_result, dict) else {}
                det = (obs.get("detail") or {}) if isinstance(obs.get("detail"), dict) else {}
                candidate = det.get("path") or obs.get("path")
                if not candidate:
                    for k, v in (action_input or {}).items():
                        if isinstance(k, str) and "path" in k.lower():
                            candidate = v
                            break
                self.last_written_file = candidate or ""
        except Exception:
            pass

        # 7) 프롬프트에 함수콜/응답 기록
        try:
            if candidate:
                self.prompt_content.append(self._as_content(candidate))
            else:
                self.prompt_content.append(
                    types.Content(
                        role="model",
                        parts=[types.Part(function_call=types.FunctionCall(name=action, args=action_input or {}))]
                    )
                )
            self.prompt_content.append(
                types.Content(
                    role="user",
                    parts=[types.Part(function_response=types.FunctionResponse(name=action, response={"result": observation_result}))]
                )
            )
        except NameError:
            pass

        # 8) 백그라운드 최적화 트리거
        try:
            import asyncio
            if not hasattr(self, "optim_manager") or self.optim_manager is None:
                from optimization_manager import OptimizationManager
                self.optim_manager = OptimizationManager()
            generate_tool_definition_and_code = globals().get("generate_tool_definition_and_code")
            register_newly_generated_tool = globals().get("register_newly_generated_tool")
            handle_reload_skills = globals().get("handle_reload_skills")
            if generate_tool_definition_and_code and register_newly_generated_tool and handle_reload_skills:
                workflow_str_list = getattr(self, "session_workflow", [])
                user_goal_for_optimization = getattr(self, "user_goal", None)
                asyncio.create_task(
                    self.optim_manager.record_and_maybe_materialize(
                        workflow_str_list,
                        user_goal_for_optimization,
                        generate_tool_definition_and_code=generate_tool_definition_and_code,
                        register_newly_generated_tool=register_newly_generated_tool,
                        handle_reload_skills=handle_reload_skills
                    )
                )
        except Exception as e:
            print(f"[Optimization Hook Error] {e}")

        # 9) 다음 사이클로 진행
        return False, thought, action, action_input, observation_result
    def _save_tool_call_log(self, iteration: int, tool_name: str, parameters: dict):
        """DB에 툴 호출 기록을 동기적으로 저장합니다."""
        try:
            new_log = ToolCallLog(
                session_id=self.session_id, request_id=self.request_id, iteration=iteration,
                tool_name=tool_name, parameters=parameters, timestamp=datetime.datetime.now()
            )
            self.db.add(new_log); self.db.commit()
            print(f"[📊 Tool Log Saved] Session: {self.session_id}, Request: {self.request_id}, Tool: {tool_name}")
        except Exception as e:
            self.db.rollback()
            print(f"[❌ Tool Log Error] 툴 로그 DB 저장 실패: {e}")
    async def stream_run(self):
        """
        에이전트의 전체 실행을 스트리밍하는 메인 제너레이터.
        """
        try:
            yield f"data: {json.dumps({'type': 'session_info', 'session_id': self.session_id})}\n\n"
            await self._initialize()

            # 1. 실행 전, 최적의 '첫 단계'를 제안받습니다.
            first_step_proposal = await self._propose_first_step()

            # 2. 메인 루프를 실행합니다.
            for i in range(MAX_AGENT_ITERATIONS):
                # 첫 번째 이터레이션에만 제안을 전달합니다.
                proposal_for_this_iteration = first_step_proposal if i == 0 else (None, None)
                
                is_final, thought, action, action_input, observation_result = await self._run_iteration(
                    i, first_step_proposal=proposal_for_this_iteration
                )
                
                if thought:
                    async for chunk in stream_text_by_char('thought_stream', thought): yield chunk
                    yield f"data: {json.dumps({'type': 'thought_stream_end'})}\n\n"
                
                if is_final:
                    obs = observation_result or {}
                    if not isinstance(obs, dict):
                        obs = {"status": "error", "reason": str(obs)}
                    final_answer = (action_input or {}).get("answer") or obs.get("answer") or "작업이 완료되었습니다."
                    if action == "error":
                        async for chunk in stream_text_by_char('error_stream', final_answer): yield chunk
                    else:
                        if len(self.session_workflow) >= 2:
                            print(f"✨ 최적화 가능성 발견! 워크플로우 길이: {len(self.session_workflow)}. 최적화 프로세스를 시작합니다.")
                            self.background_tasks.add_task(run_optimization_workflow, {"completed_workflow": self.session_workflow, "user_goal": self.user_goal})
                        
                        await asyncio.to_thread(save_message, self.session_id, self.user_goal, self.db, self.current_user_id)
                        await asyncio.to_thread(save_message, self.session_id, final_answer, self.db, None)
                        try:
                            if getattr(self, "_materialize_queue", None):
                                for spec in self._materialize_queue:
                                    # 네 쪽 최적화 매니저 API에 맞게 바꿔서 호출
                                    self.optim_manager.record_materialize_proposal(spec)
                        except Exception as _e:
                            print(f"[materialize] 제안 전달 실패: {_e}")
                        
                        async for chunk in stream_text_by_char('final_answer_stream', final_answer): yield chunk
                    
                    break

        except Exception as e:
            error_msg = f"Agent 실행 중 심각한 오류 발생: {e}"
            print(f"❌ {error_msg}")
            import traceback; traceback.print_exc()
            async for chunk in stream_text_by_char('error_stream', error_msg): yield chunk
        finally:
            self.db.close()
            print(f"[DB Session Closed] Session for {self.session_id} has been closed.")
        
        yield f"data: {json.dumps({'type': 'final_stream_end', 'lastWrittenFile': self.last_written_file})}\n\n"
        yield "data: [DONE]\n\n"
    def _load_tool_definitions(self) -> dict:
        """
        generated_definitions.json을 읽어 각 tool의 허용/필수 파라미터를 캐시.
        """
        defs_path = BACKEND_DIR / "tools" / "generated_definitions.json"
        if not defs_path.exists():
            return {}
        try:
            data = json.loads(defs_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        tools = {}
        for t in data.get("tools", []):
            name = t.get("name")
            params = t.get("parameters", {})
            props = params.get("properties", {}) if isinstance(params, dict) else {}
            required = set(params.get("required", [])) if isinstance(params, dict) else set()
            if name:
                tools[name] = {"allowed": set(props.keys()), "required": required}
        return tools
    def _looks_like_path_or_url(self, s: str) -> bool:
        if not isinstance(s, str): return False
        s = s.strip()
        if s.startswith("http://") or s.startswith("https://"):
            return True
        # 파일 경로/확장자 흔적
        if re.search(r"[\\/]", s) and re.search(r"\.[A-Za-z0-9]{1,5}($|\?)", s):
            return True
        return False
    def _build_user_message(self, text: str):
        """
        문자열 prompt를 Gemini API가 받는 messages 포맷으로 변환.
        google.genai.types 가 있으면 그것을, 없으면 dict 포맷을 사용.
        """
        try:
            from google.genai import types
            return [types.Content(role="user", parts=[types.Part(text=text)])]
        except Exception:
            return [{"role": "user", "parts": [{"text": text}]}]
    async def _select_payload_key_via_llm(self, tool_spec: dict, action_input: dict) -> str | None:
        """
        LLM에게 '메인 텍스트 페이로드'로 적합한 문자열 필드 1개를 선택시킨다.
        - 스키마(properties/description/type)와 현재 값 샘플을 보여주고
        - 파일 경로/URL/타이틀/토픽 등은 제외하도록 지시
        - JSON {"key":"..."}만 반환
        """
        params = (tool_spec or {}).get("parameters", {}) or {}
        props  = (params.get("properties") or {}) if isinstance(params.get("properties"), dict) else {}
        # 문자열 필드만 추출
        cand = {k:v for k,v in props.items() if isinstance(v, dict) and v.get("type") == "string"}
        if not cand:
            return None

        # 샘플 값과 설명 제공
        sample = {k: (action_input.get(k) if isinstance(action_input.get(k), str) else "") for k in cand.keys()}
        schema_brief = {k: {"description": cand[k].get("description",""), "type": "string"} for k in cand.keys()}

        instruction = (
            "Pick the single property key that holds the MAIN TEXT PAYLOAD to be written.\n"
            "- Exclude paths, URLs, titles, topics, or identifiers.\n"
            "- Prefer the field that contains the document/article/html/code body.\n"
            "Return JSON only: {\"key\":\"<property_name>\"}"
        )
        prompt = f"{instruction}\n\nSCHEMA:\n{json.dumps(schema_brief, ensure_ascii=False, indent=2)}\n\nSAMPLES:\n{json.dumps(sample, ensure_ascii=False, indent=2)}"

        resp, _ = await self._call_llm_safe(self._build_user_message(prompt), use_tools=False)
        text = getattr(resp, "text", None) if not isinstance(resp, str) else resp
        try:
            obj = json.loads((text or "").strip())
            key = obj.get("key")
            return key if key in cand else None
        except Exception:
            return None

    def _fallback_payload_key(self, tool_spec: dict, action_input: dict) -> str | None:
        """
        LLM 선택 실패시: 문자열 필드 중 경로/URL로 보이는 값은 제외하고,
        현재 값이 가장 긴 키를 선택. 모두 비면 임의의 첫 키.
        """
        params = (tool_spec or {}).get("parameters", {}) or {}
        props  = (params.get("properties") or {}) if isinstance(params.get("properties"), dict) else {}
        cands  = [k for k,v in props.items() if isinstance(v, dict) and v.get("type") == "string"]
        if not cands: return None

        filtered = []
        for k in cands:
            val = action_input.get(k, "")
            if isinstance(val, str) and self._looks_like_path_or_url(val):
                continue
            filtered.append(k)
        if not filtered:
            filtered = cands

        best_k, best_len = None, -1
        for k in filtered:
            v = action_input.get(k, "")
            L = len(v) if isinstance(v, str) else -1
            if L > best_len:
                best_k, best_len = k, L
        return best_k if best_k is not None else filtered[0]

    async def _enrich_preserving_format(self, current_text: str, user_goal: str, min_chars: int) -> str:
        """
        형식(HTML/Markdown/코드/평문)을 유지하며 'current_text'를 충분히 확장.
        메타멘트 없이 결과물만.
        """
        instruction = (
            "Expand the CURRENT_TEXT while strictly preserving its format/type.\n"
            "- If HTML: output valid HTML (doctype optional if already present).\n"
            "- If code: output only the code in the same language.\n"
            "- If Markdown: keep Markdown.\n"
            "- If plain text: produce a longer high-quality article in the same language.\n"
            "- DO NOT add any explanations or meta text; output content only.\n"
            f"- Target length ≥ {min_chars} characters.\n"
        )
        prompt = f"{instruction}\n\n[USER_GOAL]\n{user_goal}\n\n[CURRENT_TEXT]\n{current_text or ''}\n"
        resp, _ = await self._call_llm_safe(self._build_user_message(prompt), use_tools=False)
        text = getattr(resp, "text", None) if not isinstance(resp, str) else resp
        return (text or "").strip()

    async def _maybe_enrich_text_payload_strong(self, *, action: str, action_input: dict, tool_spec: dict, min_threshold: int, min_chars: int) -> dict:
        key = self._pick_payload_field_fast(tool_spec, action_input or {})
        if not key: return action_input
        cur = action_input.get(key, "")
        if not isinstance(cur, str) or len(cur.strip()) >= min_threshold:
            return action_input

        print(f"[ENRICH] start action={action}, key=content, len={len((action_input or {}).get('content',''))} → target≥{min_chars}")
        enriched = await self._compose_or_enrich(self.user_goal or "", cur, min_chars=min_chars)
        print(f"[ENRICH] done len={len((action_input or {}).get('content',''))}")
        if enriched and len(enriched) > len(cur):
            print(f"[ENRICH] done key={key}, len {len(cur)} → {len(enriched)}")
            new_args = dict(action_input or {})
            new_args[key] = enriched
            return new_args
        
        print(f"[ENRICH] fail/no-change key={key}, kept len={len(cur)}")
        return action_input
    def _validate_action_input(self, action: str, action_input: dict) -> tuple[bool, dict]:
        """
        호출 직전 스키마 검증.
        - unknown_keys / missing_keys 있으면 False 반환 + 힌트 포함
        """
        if not hasattr(self, "tool_definitions") or not self.tool_definitions:
            try:
                self.tool_definitions = self._load_tool_definitions()
            except Exception:
                self.tool_definitions = {}

        schema = self.tool_definitions.get(action)
        if not schema:
            # 정의 없는 도구(레거시)면 통과
            return True, {}

        allowed = schema.get("allowed", set())
        required = schema.get("required", set())

        keys = set((action_input or {}).keys())
        unknown = keys - allowed
        missing = required - keys

        if unknown or missing:
            return False, {
                "status": "schema_error",
                "unknown_keys": sorted(list(unknown)),
                "missing_keys": sorted(list(missing)),
                "hint": f"Use only keys {sorted(list(allowed))} for tool '{action}'."
            }
        return True, {}
    def _current_plan_tools(self) -> list[str]:
        tools = []
        for s in getattr(self, "session_workflow", []):
            try:
                obj = json5.loads(s)
                name = obj.get("tool_name")
                if isinstance(name, str) and name:
                    tools.append(name)
            except: 
                continue
        return tools
    def _maybe_override_with_registry_tool(self, action: str, action_input: dict):
        try:
            plan_steps = self._get_planned_steps() or [action]
            catalog = self.tool_manager.tool_catalog

            def _ensure_args(name: str, params: dict):
                try:
                    ok, norm, err = self.tool_manager.ensure_args(name, params or {})
                    return ok, (norm or {}), (err or None)
                except Exception as e:
                    return False, {}, str(e)

            pick = select_best_covering_tool(plan_steps, catalog, _ensure_args, min_cover_len=1)
            if not pick:
                return action, action_input, {"overridden": False}

            picked_tool, base_args = pick

            # content/출력 경로는 승계.
            merged = dict(base_args or {})
            if "content" in (action_input or {}):
                merged["content"] = action_input["content"]

            # ✅ file_path는 "사용자가 명시한 경우"에만 승계. (LLM이 임의 생성한 경로는 버림)
            if "file_path" in (action_input or {}):
                if self._user_explicitly_requested_path(action_input["file_path"]):
                    merged["file_path"] = action_input["file_path"]

            # ✅ output_path: 이것도 사용자 명시일 때만 승계 (LLM이 임의 생성한 경로는 무시)
            if "output_path" in (action_input or {}):
                if self._user_explicitly_requested_path(action_input["output_path"]):
                    merged["output_path"] = action_input["output_path"]

            ok, normed, err = _ensure_args(picked_tool, merged)
            if not ok:
                return action, action_input, {"overridden": False, "reason": f"ensure_args failed: {err}"}

            return picked_tool, normed, {"overridden": True, "by": "coverage"}
        except Exception as e:
            return action, action_input, {"overridden": False, "error": str(e)}
    def _coerce_json_from_text(self, raw_text: str) -> dict | None:
        """
        1) prompt_builder.extract_json_from_text() 시도
        2) 실패 시 바깥 { ... } 강탈해서 json5 파싱
        """
        try:
            from prompt_builder import extract_json_from_text
        except Exception:
            extract_json_from_text = None

        text = (raw_text or "").strip()
        if not text:
            return None

        # 1) 프로젝트 제공 파서
        if extract_json_from_text:
            try:
                cand = extract_json_from_text(text)
                if cand:
                    obj = json5.loads(cand)
                    if isinstance(obj, dict):
                        return obj
            except Exception:
                pass

        # 2) 바깥 중괄호 강탈
        try:
            l = text.find("{")
            r = text.rfind("}")
            if l != -1 and r != -1 and r > l:
                cand2 = text[l:r+1]
                obj2 = json5.loads(cand2)
                if isinstance(obj2, dict):
                    return obj2
        except Exception:
            pass

        return None
    async def _repair_json_via_llm(self, raw_text: str, max_attempts: int = 3) -> dict | None:
        """
        모델이 유효 JSON을 내놓지 못했을 때, 보정 전용 프롬프트로 최대 N회 복구.
        - 출력은 단일 JSON 객체만 허용.
        - 구조 강제: {"Thought": "...", "Action": {"tool_name": "...", "parameters": {...}}}
        """
        if not raw_text:
            return None

        REPAIR_INSTRUCTION = (
            "아래의 원본 응답을 규칙에 맞는 단일 JSON 객체로 보정하여 반환하세요.\n"
            "- 출력은 오직 JSON 객체 1개만(코드블록/주석/설명 금지)\n"
            '- 구조: {"Thought":"...","Action":{"tool_name":"...","parameters":{}}}\n'
            "- 문자열의 따옴표/줄바꿈/백틱은 JSON 문법에 맞게 이스케이프\n"
            "- 누락된 닫는 괄호/따옴표/쉼표 보정\n"
            "- parameters 안의 thought/debug/notes 등 메타키 제거\n"
            "- 이미 유효한 JSON이면 그대로 재출력\n"
        )

        for _ in range(max_attempts):
            prompt = (
                REPAIR_INSTRUCTION
                + "\n\n[원본 응답]\n<<<\n" + raw_text + "\n>>>\n"
            )
            try:
                response, _ = await self._call_llm_safe(
                    [types.Content(role="user", parts=[types.Part(text=prompt)])],
                    available_models=getattr(self, "available_models", None),
                    use_tools=False
                )
            except Exception as e:
                print(f"[JSON Repair Warning] LLM call failed: {e}")
                await asyncio.sleep(0.8)
                continue

            text = getattr(response, "text", None)
            if not text:
                await asyncio.sleep(0.4)
                continue

            obj = self._coerce_json_from_text(text)
            if isinstance(obj, dict):
                act = obj.get("Action")
                if isinstance(act, dict) and isinstance(act.get("tool_name"), str) and "parameters" in act:
                    return obj

            await asyncio.sleep(0.4)

        return None
