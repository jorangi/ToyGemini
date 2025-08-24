# tool_registry.py
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

DEFAULT_OUTPUT_PATH = "Frontend/public/longText.txt"

# 점수 가중치 (원하면 환경변수/설정으로 빼도 됨)
W_RECENCY = 0.30
W_COVERAGE = 0.25
W_SUCCESS = 0.30
W_LATENCY = 0.15

# 지수이동평균(EMA) 감쇠 계수
EMA_ALPHA = 0.25

def _safe_load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _safe_save_json(path: Path, data):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)

class ToolRegistry:
    """
    - materialized 도구 목록과 generated_definitions.json을 읽음
    - 목적 태그(purpose_tag=site_build 등)에 맞는 도구를 추천
    - 성공률, 평균 지연(ms), 최근성, 커버리지(스키마 폭)로 종합 점수화
    - 호출 후 결과를 report_result로 기록하여 다음 추천에 반영
    """
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path(__file__).parent
        self.defs_path = self.base_dir / "generated_definitions.json"
        self.state_dir = self.base_dir / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.materialized_path = self.state_dir / "materialized_tools.json"
        self.stats_path = self.state_dir / "tool_stats.json"

        self._defs = self._load_defs()
        self._mat = self._load_materialized()
        self._stats = _safe_load_json(self.stats_path, {})  # {tool_name: {calls, success, fail, avg_latency_ms, last_used_ts}}

    def refresh(self):
        self._defs = self._load_defs()
        self._mat = self._load_materialized()
        self._stats = _safe_load_json(self.stats_path, self._stats)

    # ---------- 로딩 ----------
    def _load_defs(self) -> Dict[str, Any]:
        data = _safe_load_json(self.defs_path, {})
        tools = {}
        for t in data.get("tools", []):
            name = t.get("name")
            params = t.get("parameters", {})
            props = params.get("properties", {}) if isinstance(params, dict) else {}
            required = set(params.get("required", [])) if isinstance(params, dict) else set()
            if name:
                tools[name] = {
                    "allowed": set(props.keys()),
                    "required": set(required),
                    "raw": t
                }
        return tools

    def _load_materialized(self) -> Dict[str, Any]:
        data = _safe_load_json(self.materialized_path, {})
        return data or {}

    # ---------- 조회 ----------
    def list_materialized(self) -> List[Tuple[str, Dict[str, Any]]]:
        items: List[Tuple[str, Dict, Any]] = []
        for key, info in self._mat.items():
            tool_name = info.get("tool_name")
            meta = info.get("meta", {})
            if tool_name:
                items.append((tool_name, meta))
        return items

    # ---------- 점수화 ----------
    def _coverage_score(self, tool_name: str) -> float:
        # 간단 근사치: 허용/필수 파라미터 개수로 범용성 추정 (많을수록 커버 폭이 넓다고 가정)
        d = self._defs.get(tool_name)
        if not d:
            return 0.0
        allowed = len(d["allowed"])
        required = len(d["required"])
        raw_score = allowed + max(0, 3 - required)  # 필수 적으면 가점
        # 정규화
        return min(1.0, raw_score / 10.0)

    def _recency_score(self, idx: int, total: int) -> float:
        # materialized 역순 목록에서 뒤쪽(가장 최근)에 가까울수록 점수 높게
        if total <= 1:
            return 1.0
        # idx: 0..total-1 (앞쪽이 오래된 것), 최근일수록 idx가 큼
        return idx / (total - 1)

    def _success_latency_score(self, tool_name: str) -> Tuple[float, float]:
        s = self._stats.get(tool_name, {})
        calls = max(1, int(s.get("calls", 0)))
        succ = int(s.get("success", 0))
        avg_latency = float(s.get("avg_latency_ms", 1000.0))
        success_rate = succ / calls
        # latency는 빠를수록 좋음 → 역정규화(예: <= 200ms면 1.0, 1500ms 이상이면 0.0)
        if avg_latency <= 200:
            latency_score = 1.0
        elif avg_latency >= 1500:
            latency_score = 0.0
        else:
            latency_score = 1.0 - ((avg_latency - 200) / (1500 - 200))
        return success_rate, max(0.0, min(1.0, latency_score))

    def _composite_score(self, tool_name: str, recency: float) -> float:
        cov = self._coverage_score(tool_name)
        succ, lat = self._success_latency_score(tool_name)
        score = (W_RECENCY * recency) + (W_COVERAGE * cov) + (W_SUCCESS * succ) + (W_LATENCY * lat)
        return score

    # ---------- 추천/선택 ----------
    def _dependency_tools_from_meta(self, meta: dict) -> set[str]:
        dg = (meta.get("definition") or {}).get("dependency_graph") or {}
        nodes = dg.get("nodes") or []
        return {n.get("tool_name") for n in nodes if isinstance(n, dict) and n.get("tool_name")}

    def recommend_by_coverage(self, plan_tools: list[str]) -> Optional[dict]:
        mats = self.list_materialized()
        if not mats: return None
        best, best_score = None, -1.0
        total = len(mats)
        for idx, (tool_name, meta) in enumerate(mats):
            if tool_name not in self._defs: 
                continue
            deps = self._dependency_tools_from_meta(meta)
            if not deps:
                continue
            inter = len(deps.intersection(plan_tools))
            if inter == 0:
                continue
            coverage = inter / max(1, len(deps))
            recency = self._recency_score(idx, total)
            quality = self._composite_score(tool_name, recency)
            score = 0.6 * coverage + 0.4 * quality
            if score > best_score:
                best = {"tool_name": tool_name, "schema": self._defs[tool_name], "score": score, "deps": list(deps)}
                best_score = score
        return best
    def ensure_args(self, tool_name: str, base_args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if tool_name not in self._defs:
            return None
        schema = self._defs[tool_name]
        required = schema["required"]
        allowed = schema["allowed"]
        args = dict(base_args or {})

        # 기본값 보정
        if "output_path" in allowed and "output_path" not in args:
            args["output_path"] = DEFAULT_OUTPUT_PATH
        if "topic" in required and "topic" not in args:
            title = args.get("title")
            if title:
                args["topic"] = title

        # 필수 충족 확인
        if not required.issubset(set(args.keys())):
            return None

        # 불필요 키 제거
        args = {k: v for k, v in args.items() if (k in allowed) or (k in required)}
        return args

    # ---------- 실행 결과 보고 ----------
    def report_result(self, tool_name: str, success: bool, latency_ms: float):
        if tool_name not in self._stats:
            self._stats[tool_name] = {"calls": 0, "success": 0, "fail": 0, "avg_latency_ms": float(latency_ms), "last_used_ts": time.time()}
        s = self._stats[tool_name]
        s["calls"] = int(s.get("calls", 0)) + 1
        if success:
            s["success"] = int(s.get("success", 0)) + 1
        else:
            s["fail"] = int(s.get("fail", 0)) + 1

        # EMA 업데이트
        prev = float(s.get("avg_latency_ms", latency_ms))
        s["avg_latency_ms"] = (EMA_ALPHA * float(latency_ms)) + ((1.0 - EMA_ALPHA) * prev)

        s["last_used_ts"] = time.time()
        _safe_save_json(self.stats_path, self._stats)
