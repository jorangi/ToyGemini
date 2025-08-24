# tests/run_smoke.py
import sys, pathlib
# 프로젝트 루트(Backend)를 sys.path에 추가
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 이제 패키지로 직접 임포트 (shim 불필요)
import metatools.tool_selector as ts
from agent import _adapt_action_params_for  # 전역 노출 확인

import asyncio
from types import SimpleNamespace

# 가짜 ToolManager (테스트용)
class TM:
    all_definitions = [
        SimpleNamespace(
            name="save_text",
            description="Save text to a file path",
            tags=["fs","io"],
            parameters={
                "type":"object",
                "properties":{"path":{"type":"string"},"text":{"type":"string"}},
                "required":["path","text"]
            }
        ),
        SimpleNamespace(
            name="final_response",
            description="Send final message to user",
            tags=["terminal"],
            parameters={
                "type":"object",
                "properties":{"text":{"type":"string"}},
                "required":["text"]
            }
        ),
    ]

async def main():
    tm = TM()
    # 1) 카탈로그/프리필터
    cat = ts.build_minimal_catalog(tm)
    print("[CATALOG]", [c.name for c in cat])
    pf = ts.CatalogPrefilter()
    out = pf.top_n("파일에 'hello' 저장", cat, n=2)
    print("[PREFILTER]", [c.name for c in out])

    # 2) 도구 선택
    res = await ts.decide_tool(tm, "파일에 'hello' 저장", ["write to disk"])
    print("[DECIDE]", getattr(res, "decision", None), getattr(res, "tool_name", None))

    # 3) 파라미터 제안 (선택되면)
    if getattr(res, "tool_name", None):
        params = await ts.propose_parameters(tm, "파일에 'hello' 저장", res.tool_name)
        print("[PARAMS]", params)

    # 4) STRICT 필터 확인
    def handler(path, text): return (path, text)
    filtered = _adapt_action_params_for(handler, "save_text", {"path":"E:/tmp/hello.txt","text":"hi","extra":1})
    print("[STRICT]", filtered)

if __name__ == "__main__":
    asyncio.run(main())
