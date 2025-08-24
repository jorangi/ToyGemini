# metatools/tool_selector.py
from __future__ import annotations
from metatools.tool_selector import *

"""
Smart Tool Selector (catalog-first, coverage-greedy)
---------------------------------------------------
- Build a minimal tool catalog from tool_manager.
- Prefilter catalog by semantic similarity if SentenceTransformer is available;
  otherwise, DO NOT use any heuristic → pass-through top-N (LLM-only selection).
- Ask the LLM to SELECT exactly one tool maximizing coverage of remaining steps,
  or respond with 'need_more' (alpha-slice only), or 'materialize' (propose macro tool).
- Then, generate parameters that match tool schema exactly (no alias/auto-mapping).
- Absolutely no keyword-based slicing or heuristic embedders.

Design notes (cycle safety):
- Do NOT import router/agent modules at import-time.
- LLM call helper lazily imports llm_utils.call_gemini_agent inside functions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import json, json5
import math, re, unicodedata
import importlib, sys
_mod = importlib.import_module("metatools.tool_selector")
sys.modules[__name__] = _mod
from typing import List, Dict, Tuple, Optional, Callable

# Optional semantic embedding backend
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    import numpy as np  # type: ignore
    _HAS_ST = True
except Exception:
    SentenceTransformer = None  # type: ignore
    np = None  # type: ignore
    _HAS_ST = False


def _lcs_length(seq_a: List[str], seq_b: List[str]) -> int:
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return 0
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            tmp = dp[j]
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[m]

def _extract_pattern(tool_spec: Dict) -> List[str]:
    graph = (tool_spec or {}).get("dependency_graph") or {}
    nodes = graph.get("nodes") or []
    out = []
    for node in nodes:
        name = (node or {}).get("tool_name")
        if name:
            out.append(name)
    return out

def select_best_covering_tool(
    plan_steps: List[str],
    tool_catalog: Dict[str, Dict],
    ensure_args: Callable[[str, Dict], Tuple[bool, Dict, Optional[str]]],
    min_cover_len: int = 1  # ✅ 단일 스텝도 허용
) -> Optional[Tuple[str, Dict]]:
    """
    - 키워드 없이 plan_steps vs dependency_graph LCS로 최적 도구 선택
    - required 인자는 ensure_args로 주입 가능해야 함
    - 동률일 때 '더 긴 패턴(=더 많은 절차)'을 가진 도구를 선호 → 매크로 선호
    """
    if not plan_steps:
        return None

    best = None  # (score_tuple, tool_name, args)
    total = len(plan_steps)

    for tname, spec in (tool_catalog or {}).items():
        pattern = _extract_pattern(spec)
        if len(pattern) < min_cover_len:
            continue

        cover = _lcs_length(plan_steps, pattern)
        if cover < min_cover_len:
            continue

        ratio = cover / max(1, total)

        # 실행 가능성: 기본값 주입 테스트
        ok, norm_args, err = ensure_args(tname, {})
        if not ok:
            # 힌트 인자(일반 키)로 재시도
            hint = {"content": "", "file_path": "", "output_path": ""}
            ok, norm_args, err = ensure_args(tname, hint)
            if not ok:
                continue

        # 점수: 1) 커버 길이 2) 커버 비율 3) 패턴 길이(긴 패턴 선호=매크로 선호)
        score_tuple = (cover, ratio, len(pattern))
        if best is None or score_tuple > best[0]:
            best = (score_tuple, tname, norm_args or {})

    if best is None:
        return None
    _, tool_name, args = best
    return tool_name, args
# =============================
# Utilities
# =============================
async def classify_goal_with_llm(
    user_goal: str,
    available_models: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    LLM에게 사용자 요청의 의도를 분류시킨다.
    - 'system': 파일/DB/명령/도구 실행/코드 수정 등 시스템 작업
    - 'narrative': TRPG/시나리오/롤플레잉/스토리텔링
    반환 형식: {"intent": "system"|"narrative", "confidence": float}
    """
    prompt = (
        "당신은 분류기입니다. 다음 사용자 요청의 의도를 분류하세요.\n"
        "- 'system': 파일/DB/도구 실행/코드 수정/명령 수행 등 시스템 작업\n"
        "- 'narrative': TRPG/시나리오/롤플레잉/스토리텔링\n"
        "JSON만 출력:\n"
        '{ "intent": "system" | "narrative", "confidence": 0.0 }\n'
        f"\n요청: {user_goal}\n"
    )
    obj = await _call_llm_json(prompt, available_models=available_models)
    intent = (obj or {}).get("intent")
    conf = (obj or {}).get("confidence", 0.5)

    # LLM이 엉뚱한 값을 내면 보수적 기본값으로 수습
    if intent not in ("system", "narrative"):
        intent = "system"
    try:
        conf = float(conf)
    except Exception:
        conf = 0.5
    return {"intent": intent, "confidence": conf}

def _normalize_text(s: Any) -> str:
    if not isinstance(s, str):
        return ""
    return unicodedata.normalize("NFKC", s).strip()

def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        try:
            return json5.dumps(obj)
        except Exception:
            return str(obj)

def _extract_json_block(txt: str) -> Optional[dict]:
    if not txt:
        return None
    t = txt.strip()
    # direct try
    for parser in (json5.loads, json.loads):
        try:
            if t.startswith("{") and t.endswith("}"):
                return parser(t)
        except Exception:
            pass
    # find first {...}
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        sub = t[start:end+1]
        for parser in (json5.loads, json.loads):
            try:
                return parser(sub)
            except Exception:
                pass
    return None


# =============================
# Data structures
# =============================
from dataclasses import dataclass, field
from typing import Any, Dict, List
@dataclass
class CatalogItem:
    name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # ⬇️ 테스트가 요구하는 필드들 (스키마 & 커버리지 그래프)
    schema: Dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}, "required": []})
    dependency_graph: List[List[int]] = field(default_factory=list)

    # ⬇️ 기존 코드가 참조할 수도 있는 패널티 필드들(있어도/없어도 무방, 기본 0.0)
    safety_penalty: float = 0.0
    cost_penalty: float = 0.0
    risk_penalty: float = 0.0
# =============================
# Catalog building
# =============================

def build_minimal_catalog(tool_manager) -> List[CatalogItem]:
    """
    Build minimal catalog from tool_manager.all_definitions
    Deterministic order recommended (definition order assumed).
    """
    items: List[CatalogItem] = []
    for d in getattr(tool_manager, "all_definitions", []) or []:
        name = _normalize_text(getattr(d, "name", ""))
        desc = _normalize_text(getattr(d, "description", ""))
        if not name:
            continue
        # optional tags — if you have a tags source, inject here
        tags = list(getattr(d, "tags", []) or [])
        items.append(CatalogItem(name=name, description=desc, tags=tags))

    # dedup by name, keep first
    seen = set()
    out: List[CatalogItem] = []
    for it in items:
        if it.name in seen:
            continue
        seen.add(it.name)
        out.append(it)
    return out


# =============================
# Prefilter (semantic-only)
# =============================

def _cosine(a, b) -> float:
    if np is None:
        dot = sum(x*y for x, y in zip(a, b))
        na = math.sqrt(sum(x*x for x in a)) or 1.0
        nb = math.sqrt(sum(x*x for x in b)) or 1.0
        return dot / (na*nb)
    denom = (float(np.linalg.norm(a)) * float(np.linalg.norm(b))) or 1.0
    return float(np.dot(a, b) / denom)

class CatalogPrefilter:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = None
        if _HAS_ST:
            try:
                self.embedder = SentenceTransformer(model_name)  # type: ignore
            except Exception:
                self.embedder = None

    def top_n(self, user_goal: str, catalog: List[CatalogItem], n: int = 60) -> List[CatalogItem]:
        # pass-through if no embedder
        if not catalog:
            return []
        limit = max(1, min(n, len(catalog)))
        if self.embedder is None:
            return catalog[:limit]

        docs = [f"{c.name}\n{c.description}\n{','.join(c.tags)}" for c in catalog]
        vecs = self.embedder.encode([user_goal] + docs)  # type: ignore
        q = vecs[0]; doc_vecs = vecs[1:]
        scores: List[Tuple[int, float]] = []
        for idx, v in enumerate(doc_vecs):
            scores.append((idx, _cosine(q, v)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [catalog[i] for i, _ in scores[:limit]]


# =============================
# Schema helpers
# =============================

def _schema_to_dict(schema_obj: Any) -> Dict[str, Any]:
    if schema_obj is None:
        return {"type": "object", "properties": {}, "required": []}
    if isinstance(schema_obj, dict):
        return schema_obj
    try:
        t = getattr(schema_obj, "type", "object")
        props = getattr(schema_obj, "properties", {}) or {}
        req = getattr(schema_obj, "required", []) or []
        d: Dict[str, Any] = {"type": t if isinstance(t, str) else str(t), "properties": {}, "required": []}
        if isinstance(props, dict):
            for k, v in props.items():
                d["properties"][k] = v if isinstance(v, dict) else _schema_to_dict(v)
        if isinstance(req, (list, tuple, set)):
            d["required"] = list(req)
        return d
    except Exception:
        return {"type": "object", "properties": {}, "required": []}

def get_schema_for_tool(tool_manager, tool_name: str) -> Optional[Dict[str, Any]]:
    for d in getattr(tool_manager, "all_definitions", []) or []:
        if getattr(d, "name", None) == tool_name:
            schema_obj = getattr(d, "parameters", None)
            return _schema_to_dict(schema_obj)
    return None


# =============================
# Prompt builders
# =============================

def build_decision_prompt(user_goal: str, remaining_steps: List[str], catalog_slice: List[CatalogItem]) -> str:
    cat = [dict(name=x.name, description=x.description, tags=x.tags) for x in catalog_slice]
    return (
        "당신은 '도구 선택기'입니다. 아래 카탈로그에서 **이번 한 번의 호출**로 "
        "**가장 많은 원자 단계를 안전하게 처리**할 수 있는 **단일 도구 1개**를 선택하세요.\n"
        "규칙:\n"
        "- 종료성 도구(final_response 등)는 남은 단계가 0일 때만 사용 가능. steps에는 포함하지 마세요.\n"
        "- 남은 단계가 있을 때는 종료성 도구(final_response 등)를 선택하지 말 것.\n"
        "- 출력은 JSON만.\n"
        "- Coverage(커버 단계 수) > Safety(파괴적 작업 패널티) > Cost(토큰/호출) > Risk(불명확성) 순.\n"
        "- 고를 수 없으면 'need_more'로 응답하고, 추가 보고 싶은 range(A-C|C-K|K-Z)를 지정.\n"
        "- 정말 적합한 것이 없고 반복적으로 나타나는 패턴이면 'materialize'로 새 도구 스펙을 간단 제안.\n"
        "\n[사용자 목표]\n"
        f"{user_goal}\n"
        "\n[남은 단계]\n"
        f"{_safe_json(remaining_steps)}\n"
        "\n[카탈로그]\n"
        f"{_safe_json(cat)}\n"
        "\n[출력 — JSON only]\n"
        "{\n"
        '  "decision": "select" | "need_more" | "materialize" | "none",\n'
        '  "tool_name": "도구명",\n'
        '  "covers_steps": [0,2,3],\n'
        '  "remaining_steps": [1,4],\n'
        '  "reason": "간단 근거",\n'
        '  "request": {"range": "A-C|C-K|K-Z" | null},\n'
        '  "proposed_tool": {"name":"...","description":"...","tags":["macro"],"covers_steps":[...],'
        '"io_contract":{"inputs":["..."],"outputs":["..."],"preconditions":["..."],"postconditions":["..."]}}\n'
        "}\n"
    )

def build_params_prompt(user_goal: str, tool_name: str, schema: Dict[str, Any]) -> str:
    return (
        "당신은 '도구 입력 설계자'입니다. 아래 선택된 도구의 **parameters**만 작성하세요.\n"
        "규칙:\n"
        "- 스키마의 allowed/required 키와 **정확히 일치**해야 합니다. 동의어/별칭/자동치환 금지.\n"
        "- 알 수 없는 키 금지. 타입 불일치 금지. JSON만 출력.\n"
        "\n[도구]\n"
        f"name: {tool_name}\n"
        f"schema: {_safe_json(schema)}\n"
        f"goal: {user_goal}\n"
        "\n[출력 — JSON only]\n"
        '{ "parameters": { /* schema에 부합 */ } }\n'
    )


# =============================
# LLM bridge (lazy import)
# =============================

async def _call_llm_json(prompt: str, available_models: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Lazily import llm_utils.call_gemini_agent to avoid circular imports.
    Accepts either (prompt, ...) or (messages=[...], ...), returns dict.
    """
    # lazy import here
    try:
        from llm_utils import call_gemini_agent  # type: ignore
    except Exception:
        async def call_gemini_agent(*args, **kwargs):  # type: ignore
            class _Dummy: text = "{}"; content = "{}"
            return _Dummy(), {}
    # try call with prompt
    try:
        resp, _ = await call_gemini_agent(prompt, available_models=available_models)  # type: ignore
        text = getattr(resp, "content", None) or getattr(resp, "text", "") or ""
        obj = _extract_json_block(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    # fallback: try messages style
    try:
        messages = [dict(role="user", parts=[dict(text=prompt)])]
        resp, _ = await call_gemini_agent(messages, available_models=available_models)  # type: ignore
        text = getattr(resp, "content", None) or getattr(resp, "text", "") or ""
        obj = _extract_json_block(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


# =============================
# Expansion helper (alpha only)
# =============================

def _alpha_slice(catalog: List[CatalogItem], rng: str) -> List[CatalogItem]:
    """
    rng: "A-C", "C-K", "K-Z" (inclusive). Non-alpha heads go to last bucket if end == 'Z'.
    """
    m = re.match(r"^\s*([A-Za-z])\s*-\s*([A-Za-z])\s*$", (rng or ""))
    if not m:
        return catalog
    a, b = m.group(1).upper(), m.group(2).upper()
    if a > b:
        a, b = b, a
    out: List[CatalogItem] = []
    for it in catalog:
        head = (it.name or "").strip()[:1].upper()
        if not head or not head.isalpha():
            if b == "Z":
                out.append(it)
            continue
        if a <= head <= b:
            out.append(it)
    return out


# =============================
# Public API
# =============================
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class DecisionResult:
    decision: str
    tool_name: Optional[str] = None
    covers_steps: Optional[List[int]] = None
    remaining_steps: Optional[List[int]] = None
    reason: Optional[str] = None
    request: Optional[Dict[str, Any]] = None
    proposed_tool: Optional[Dict[str, Any]] = None
    
async def decide_tool(
    tool_manager,
    user_goal: str,
    remaining_steps: List[str],
    available_models: Optional[List[str]] = None,
    top_n: int = 60,
    allow_expand: bool = True,
) -> DecisionResult:
    """
    1) Build minimal catalog
    2) Prefilter to top_n
    3) Ask LLM to pick one tool; or need_more (alpha-slice); or materialize
    """
    full_cat = build_minimal_catalog(tool_manager)
    pre = CatalogPrefilter().top_n(user_goal, full_cat, n=top_n)

    # Ban terminal tools when there are remaining steps
    catalog_names = {it.name for it in full_cat}
    _BANNABLE = {"final_response"}
    banned = {n for n in catalog_names if n in _BANNABLE}
    if remaining_steps:
        pre = [it for it in pre if it.name not in banned]

    prompt = build_decision_prompt(user_goal, remaining_steps, pre)
    obj = await _call_llm_json(prompt, available_models=available_models)
    decision = (obj.get("decision") or "").strip().lower() if isinstance(obj, dict) else ""

    if decision == "select" and isinstance(obj.get("tool_name"), str):
        return DecisionResult(
            decision="select",
            tool_name=obj.get("tool_name"),
            covers_steps=obj.get("covers_steps") if isinstance(obj.get("covers_steps"), list) else None,
            remaining_steps=obj.get("remaining_steps") if isinstance(obj.get("remaining_steps"), list) else None,
            reason=obj.get("reason"),
        )

    if decision == "materialize":
        return DecisionResult(
            decision="materialize",
            proposed_tool=obj.get("proposed_tool") if isinstance(obj.get("proposed_tool"), dict) else None,
            reason=obj.get("reason"),
        )

    if decision == "need_more" and allow_expand:
        req = obj.get("request") if isinstance(obj, dict) else {}
        rng = req.get("range") if isinstance(req, dict) else None
        expanded = pre
        if remaining_steps:
            expanded = [it for it in expanded if it.name not in banned]
        if isinstance(rng, str) and rng.strip():
            expanded = _alpha_slice(full_cat, rng.strip())
        if len(expanded) > max(100, top_n):
            expanded = CatalogPrefilter().top_n(user_goal, expanded, n=top_n)

        prompt2 = build_decision_prompt(user_goal, remaining_steps, expanded)
        obj2 = await _call_llm_json(prompt2, available_models=available_models)
        decision2 = (obj2.get("decision") or "").strip().lower() if isinstance(obj2, dict) else ""
        if decision2 == "select" and isinstance(obj2.get("tool_name"), str):
            return DecisionResult(
                decision="select",
                tool_name=obj2.get("tool_name"),
                covers_steps=obj2.get("covers_steps") if isinstance(obj2.get("covers_steps"), list) else None,
                remaining_steps=obj2.get("remaining_steps") if isinstance(obj2.get("remaining_steps"), list) else None,
                reason=obj2.get("reason"),
            )
        if decision2 == "materialize":
            return DecisionResult(
                decision="materialize",
                proposed_tool=obj2.get("proposed_tool") if isinstance(obj2.get("proposed_tool"), dict) else None,
                reason=obj2.get("reason"),
            )

    return DecisionResult(decision="none", reason=obj.get("reason") if isinstance(obj, dict) else "no JSON")


async def propose_parameters(
    tool_manager,
    user_goal: str,
    tool_name: str,
    available_models: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Generate only 'parameters' that strictly match the tool's schema.
    Returns {} on failure; never raises.
    """
    schema = get_schema_for_tool(tool_manager, tool_name) or {"type": "object", "properties": {}, "required": []}
    prompt = build_params_prompt(user_goal, tool_name, schema)
    obj = await _call_llm_json(prompt, available_models=available_models)
    params = obj.get("parameters") if isinstance(obj, dict) else None
    return params if isinstance(params, dict) else {}
