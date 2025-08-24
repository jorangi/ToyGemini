# tests/test_strictness.py
import asyncio
import types

def test_prefilter_pass_through_without_st(monkeypatch):
    import metatools.tool_selector as ts

    # ST 비설치 환경 시 pass-through 동작 확인
    monkeypatch.setattr(ts, "_HAS_ST", False, raising=False)
    pf = ts.CatalogPrefilter()
    cat = [
        ts.CatalogItem(name=f"T{i}", description="", tags=[], schema={}, dependency_graph=[])
        for i in range(5)
    ]
    out = pf.top_n("any goal", cat, n=3)
    assert [it.name for it in out] == ["T0", "T1", "T2"]

def test_alpha_slice_ranges():
    import metatools.tool_selector as ts
    cat = [
        ts.CatalogItem(name="Alpha", description="", tags=[], schema={}, dependency_graph=[]),
        ts.CatalogItem(name="Beta", description="", tags=[], schema={}, dependency_graph=[]),
        ts.CatalogItem(name="Kappa", description="", tags=[], schema={}, dependency_graph=[]),
        ts.CatalogItem(name="Zeta", description="", tags=[], schema={}, dependency_graph=[]),
        ts.CatalogItem(name="_misc", description="", tags=[], schema={}, dependency_graph=[]),
    ]
    # A-C면 Alpha, Beta
    ac = ts._alpha_slice(cat, "A-C")
    assert [c.name for c in ac] == ["Alpha", "Beta"]
    # K-Z면 Kappa, Zeta + 비알파벳 헤더(_misc) 포함
    kz = ts._alpha_slice(cat, "K-Z")
    assert [c.name for c in kz] == ["Kappa", "Zeta", "_misc"]

def test_strict_action_handlers_filters_unknown_keys():
    from config import StrictActionHandlers

    # 대상 함수: 허용 키는 a,b 뿐
    def f(a, b):
        return a, b

    handlers = StrictActionHandlers({"echo": f})

    async def run():
        return await handlers["echo"](a=1, b=2, c=3, d=4)  # c,d는 드롭되어야 함

    out = asyncio.run(run())
    assert out == (1, 2)  # unknown 키들이 필터링됨

def test_adapt_action_params_for_strict_drop():
    from agent import _adapt_action_params_for

    def handler(x, y):
        return x + y

    params = {"x": 10, "y": 20, "z": 999}
    out = _adapt_action_params_for(handler, "sum", params)
    assert out == {"x": 10, "y": 20}  # z는 제거

def test_preflight_signature_exact_match_ok_and_fail():
    from preflight_enforcer import assert_signature_matches_schema

    def good(a, b=1):
        return a + b

    # allowed == 실제 시그니처 키 집합, required ⊆ allowed
    assert_signature_matches_schema(good, {"a", "b"}, {"a"})

    # **kwargs 금지 확인
    def bad_kwargs(a, **kwargs):
        return a

    raised = False
    try:
        assert_signature_matches_schema(bad_kwargs, {"a"}, {"a"})
    except Exception:
        raised = True
    assert raised, "**kwargs 허용 함수는 통과하면 안 됨"

    # 시그니처-스키마 키 집합 불일치 확인
    def only_a(a):
        return a

    raised = False
    try:
        assert_signature_matches_schema(only_a, {"a", "b"}, {"a"})
    except Exception:
        raised = True
    assert raised, "스키마/시그니처 키 집합 불일치는 실패해야 함"
