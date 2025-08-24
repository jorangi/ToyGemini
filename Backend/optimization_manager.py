# optimization_manager.py
from __future__ import annotations
import json
import os
import asyncio
import json5
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from config import tool_manager, schema_to_dict

# 디스크에 패턴 카운트와 materialized 도구 정보를 저장해 재시작 후에도 유지
_STATE_DIR = Path(__file__).parent / "state"
_STATE_DIR.mkdir(exist_ok=True, parents=True)
_COUNTS_PATH = _STATE_DIR / "pattern_counts.json"
_MATERIALIZED_PATH = _STATE_DIR / "materialized_tools.json"
try:
    _STATE_DIR = _MATERIALIZED_PATH.parent  # 이미 정의돼 있으면 이 줄은 건너뜀
except NameError:
    from pathlib import Path
    _STATE_DIR = Path(__file__).resolve().parent / "state"

_PROPOSALS_PATH = _STATE_DIR / "materialize_proposals.json"

DEFAULT_THRESHOLD = int(os.environ.get("MATERIALIZE_THRESHOLD", "3"))

import hashlib, json
def _capability_hash_from_definition(defn: dict) -> str:
    """
    같은 기능을 갖는 매크로면 '항상 동일한 해시'가 나오도록 정규화해서 해싱.
    - dependency_graph.nodes[].tool_name (정렬)
    - parameters.properties 키(정렬) + required(정렬)
    - 선택: side_effects 같은 플래그도 있으면 포함
    """
    dep = (defn or {}).get("dependency_graph", {}) or {}
    nodes = dep.get("nodes") or []
    covered = sorted([
        n.get("tool_name") for n in nodes
        if isinstance(n, dict) and n.get("tool_name")
    ])
    params = (defn or {}).get("parameters", {}) or {}
    # types.Schema일 수 있으니 dict로 정규화
    try:
        from config import schema_to_dict
        params = schema_to_dict(params) if not isinstance(params, dict) else params
    except Exception:
        if not isinstance(params, dict): params = {}
    properties = sorted(list((params.get("properties") or {}).keys()))
    required = sorted(list(params.get("required") or []))
    meta = {
        "covered": covered,
        "properties": properties,
        "required": required,
    }
    s = json.dumps(meta, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
from pathlib import Path
_CANON_REG_PATH = Path(__file__).resolve().parent / "state" / "canonical_registry.json"
def _load_canon_registry() -> dict:
    try:
        if _CANON_REG_PATH.exists():
            import json5
            return json5.loads(_CANON_REG_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        pass
    return {}
def _save_canon_registry(reg: dict):
    _CANON_REG_PATH.parent.mkdir(parents=True, exist_ok=True)
    import json
    _CANON_REG_PATH.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _save_json(path: Path, data):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)

def _make_pattern_key(workflow_str_list: List[str]) -> str:
    """
    completed_workflow(문자열 JSON들)에서 tool_name만 뽑아 정렬 후 키 생성.
    예: ["{\"tool_name\":\"plan_and_discover\"}", "{\"tool_name\":\"backup_and_write\"}", ...]
    -> "backup_and_write;plan_and_discover;write_file"
    """
    try:
        names = [json5.loads(item)["tool_name"] for item in workflow_str_list]
        names_sorted = sorted(names)
        return ";".join(names_sorted)
    except Exception:
        # 실패 시 원본을 정렬 문자열로 사용 (최소한 충돌 확률 낮춤)
        safe_sorted = sorted(workflow_str_list)
        return ";".join(safe_sorted)

class OptimizationManager:
    def __init__(self, threshold: int = DEFAULT_THRESHOLD):
        self._mtx = getattr(self, "_mtx", None) or asyncio.Lock()
        self.threshold = threshold
        self._counts: Dict[str, int] = _load_json(_COUNTS_PATH, {})
        self._materialized: Dict[str, Dict[str, Any]] = _load_json(_MATERIALIZED_PATH, {})
        try:
            from config import tool_manager
            stale = []
            for k, v in dict(self._materialized).items():
                tname = (v or {}).get("tool_name")
                exists = any(getattr(d, "name", None) == tname for d in getattr(tool_manager, "all_definitions", []))
                if not exists:
                    stale.append(k)
            for k in stale:
                self._materialized.pop(k, None)
            if stale:
                _save_json(_MATERIALIZED_PATH, self._materialized)
        except Exception:
            pass
    def _build_defs_index_from_tool_manager():
        defs_index = {}
        for fd in getattr(tool_manager, "all_definitions", []):
            name = getattr(fd, "name", None)
            if not name:
                continue
            sd = schema_to_dict(getattr(fd, "parameters", None))
            props = set((sd.get("properties") or {}).keys())
            req = set(sd.get("required") or [])
            defs_index[name] = {"allowed": props, "required": req}
        return defs_index

    def _build_param_inventory_json(defs_index: dict) -> str:
        # LLM에 보여줄 “허용 키/필수 키 인벤토리” JSON (간결형)
        inv = {}
        for name, spec in defs_index.items():
            inv[name] = {
                "required": sorted(list(spec["required"])),
                "properties": sorted(list(spec["allowed"])),
            }
        import json
        return json.dumps(inv, ensure_ascii=False, indent=2)
    def record_materialize_proposal(self, spec: Dict[str, Any]):
        # 지금은 제안만 스킵(필요해지면 큐/로그로 연결)
        try:
            log = _STATE_DIR / "materialize_proposals.json"
            arr = []
            if log.exists():
                import json
                arr = json.loads(log.read_text(encoding="utf-8"))
            arr.append(spec)
            log.write_text(json.dumps(arr, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
    def get_count(self, pattern_key: str) -> int:
        return int(self._counts.get(pattern_key, 0))

    def incr_count(self, pattern_key: str) -> int:
        new_val = self.get_count(pattern_key) + 1
        self._counts[pattern_key] = new_val
        _save_json(_COUNTS_PATH, self._counts)
        return new_val

    def mark_materialized(self, pattern_key: str, tool_name: str, meta: Dict[str, Any]):
        self._materialized[pattern_key] = {"tool_name": tool_name, "meta": meta}
        _save_json(_MATERIALIZED_PATH, self._materialized)

    def is_materialized(self, pattern_key: str) -> bool:
        return pattern_key in self._materialized

    def get_materialized(self, pattern_key: str) -> Optional[Dict[str, Any]]:
        return self._materialized.get(pattern_key)

    async def record_and_maybe_materialize(
        self,
        workflow_str_list: List[str],
        user_goal_for_optimization: Optional[str],
        *,
        generate_tool_definition_and_code,   # callable
        register_newly_generated_tool,       # callable
        handle_reload_skills                 # callable
    ) -> Dict[str, Any]:
        """
        1) 패턴 키 만들고 카운트 증가
        2) 임계값 도달 && 아직 materialize 안 됨 -> 도구 생성/등록/리로드
        3) 결과(카운트, materialized 여부, 도구명 등) 리턴
        """
        if not workflow_str_list or len(workflow_str_list) < 2:
            return {"status": "skipped", "reason": "workflow_too_short"}

        pattern_key = _make_pattern_key(workflow_str_list)
        # backup_file만 잇달아 나오는 반복 패턴은 최적화 스킵(노이즈 제거)
        try:
            import json5 as _json5
            last_tools = []
            for s in workflow_str_list[-8:]:
                obj = _json5.loads(s)
                last_tools.append(obj.get("tool_name"))
            if last_tools and all(t == "backup_file" for t in last_tools):
                return {"status": "skipped", "reason": "only_backup_file_repeats", "pattern_key": pattern_key, "count": self._counts.get(pattern_key, 0)}
        except Exception:
            pass

        count = self.incr_count(pattern_key)
        if count < self.threshold:
            return {
                "status": "counted",
                "pattern_key": pattern_key,
                "count": count,
                "threshold": self.threshold,
                "materialized": False
            }

        result = {
            "status": "counted",
            "pattern_key": pattern_key,
            "count": count,
            "threshold": self.threshold,
            "materialized": False
        }

        if self.is_materialized(pattern_key):
            info = self.get_materialized(pattern_key) or {}
            tool_name = info.get("tool_name")

            from config import tool_manager
            exists = any(getattr(d, "name", None) == tool_name for d in getattr(tool_manager, "all_definitions", []))

            if not exists:
                # 상태만 남은 유령 → 상태 정리 후 생성 루트로 진행
                try:
                    self._materialized.pop(pattern_key, None)
                    _save_json(_MATERIALIZED_PATH, self._materialized)
                except Exception:
                    pass
            else:
                return {
                    "status": "already_materialized",
                    "pattern_key": pattern_key,
                    "count": count,
                    "threshold": self.threshold,
                    "materialized": True,
                    "tool_name": tool_name
                }

            # 실제 런타임/정의에 도구가 존재하는지 검증
            try:
                from config import tool_manager
                exists = any(getattr(d, "name", None) == tool_name for d in getattr(tool_manager, "all_definitions", []))
            except Exception:
                exists = False

            if not exists:
                # 상태파일만 남은 '유령' → 상태를 지우고 계속(materialize 재시도 경로로 진입)
                try:
                    self._materialized.pop(pattern_key, None)
                    _save_json(_MATERIALIZED_PATH, self._materialized)
                except Exception:
                    pass
            else:
                # 진짜로 존재 → 기존 동작대로 조기 리턴
                result.update({"materialized": True, "tool_name": tool_name, "meta": info.get("meta", {})})
                # (가독성 위해 status도 명확히)
                result["status"] = "already_materialized"
                return result

        if count < self.threshold:
            return result  # 아직 임계 미도달

        # 1) 임계 도달 → 우선 명세/코드 생성
        try:
            async with self._mtx:
                definition, code = await generate_tool_definition_and_code(
                    pattern_key, user_goal_for_optimization
                )
        except Exception as e:
            return {
                "status": "error",
                "reason": f"generate failed: {e}",
                "pattern_key": pattern_key,
                "count": count
            }

        # 2) (명세 생성 직후) 중복 검사: 이미 동일 능력 매크로 존재 시 재사용하고 등록/리로드 생략
        # 정의가 생긴 후 캐논 중복체크
        canon = _load_canon_registry()
        cap_hash = _capability_hash_from_definition(definition)
        if cap_hash in canon:
            existing_name = canon[cap_hash]["tool_name"]
            return {"status": "already_materialized", "materialized": True, "tool_name": existing_name, "pattern_key": pattern_key, "count": count}

        # 3) 신규일 때만 등록 및 스킬 리로드
        try:
            await register_newly_generated_tool(definition, code)
            await handle_reload_skills({})
        except Exception as e:
            return {
                "status": "error",
                "reason": f"register/reload failed: {e}",
                "pattern_key": pattern_key,
                "count": count
            }

        new_tool_name = definition.get("new_tool_name") or definition.get("name") or "unknown_tool"
        safe_meta = {"definition": definition}
        try:
            # 코드가 너무 크면 생략 or 첫 몇 KB만 저장
            if isinstance(code, str) and len(code) <= 200_000:
                safe_meta["code"] = code
        except Exception:
            pass

        # 4) 캐논 레지스트리에 저장(차후 중복 감지용)
        try:
            if cap_hash:
                canon = _load_canon_registry()
                canon[cap_hash] = {"tool_name": new_tool_name}
                _save_canon_registry(canon)
        except Exception:
            # 캐논 저장 실패는 치명적이지 않으니 로깅만
            print("[Optimization] warning: failed to update canonical_registry")

        self.mark_materialized(pattern_key, new_tool_name, safe_meta)

        result.update({"status": "materialized", "materialized": True, "tool_name": new_tool_name})
        return result
