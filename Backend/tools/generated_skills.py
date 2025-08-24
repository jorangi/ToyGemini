from __future__ import annotations
from pathlib import Path
from typing import Optional

try:
    from tool_registry import DEFAULT_OUTPUT_PATH
except Exception:
    DEFAULT_OUTPUT_PATH = "Frontend/public/longText.txt"

async def manage_web_content(action_handlers, *, content: str,
    file_path: Optional[str] = None,
    output_path: Optional[str] = None):
    target = file_path or output_path or DEFAULT_OUTPUT_PATH
    Path(target).parent.mkdir(parents=True, exist_ok=True)

    backup_result = await action_handlers["backup_file"](file_path=target)
    write_result  = await action_handlers["write_file"](file_path=target, content=content)

    return {
        "status": "ok",
        "detail": {
            "path": target,
            "backup": backup_result,
            "write":  write_result
        }
    }
