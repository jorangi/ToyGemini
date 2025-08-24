import asyncio
import httpx
import datetime
from pathlib import Path
from sql_router import execute_raw_sql, get_db_schema, get_db_schema_for_tables
from commands_router import execute_shell, CommandRequest
import shutil
import time
import os
import locale


# 프로젝트 절대 경로가 필요하면 사용하고, 아니면 제거해도 됩니다.
ROOT_DIR = Path("E:/Develop/ToyGemini")


# -------------------------
# 파일/콘텐츠 관련
# -------------------------
async def handle_write_file(action_input):
    """
    외부 스키마 표준 키: file_path, content
    - 과거/외부 API가 'filepath'를 요구하면 핸들러에서 변환
    반환 형식 통일:
      성공: {"status":"ok", "path":<str>, "bytes_written":<int>|None, "detail":<any>}
      실패: {"status":"error", "reason":<str>}
    """
    # 표준 키 우선(file_path), 백워드 호환(filepath)
    file_path = action_input.get("file_path") or action_input.get("filepath")
    content = action_input.get("content")
    if not file_path or content is None:
        return {
            "status": "error",
            "reason": "missing_required_keys: file_path/content"
        }

    try:
        async with httpx.AsyncClient() as client:
            # 외부 API 스펙이 'filepath'라면 여기서 변환
            resp = await client.post(
                "http://localhost:8000/write-file",  # 실제 서버 주소/포트에 맞춰 조정
                json={"filepath": file_path, "content": content},
                timeout=5
            )
        if resp.status_code == 200:
            data = resp.json()
            # 외부 응답을 우리 표준으로 정규화
            return {
                "status": "ok",
                "path": file_path,
                "bytes_written": data.get("bytes_written"),
                "detail": data
            }
        else:
            return {
                "status": "error",
                "reason": f"write-file API {resp.status_code}: {resp.text}"
            }
    except Exception as e:
        return {
            "status": "error",
            "reason": f"write-file API call failed: {e}"
        }


# -------------------------
# DB 관련

try:
    from config import ROOT_DIR as CFG_ROOT_DIR  # 프로젝트 루트
except Exception:
    CFG_ROOT_DIR = None

try:
    from config import BACKEND_DIR as CFG_BACKEND_DIR
except Exception:
    CFG_BACKEND_DIR = None

# (기존에 있다면 유지) 하드코딩된 루트가 있다면 우선 후보로 사용
try:
    ROOT_DIR  # noqa
except NameError:
    ROOT_DIR = None

# -------------------------
async def handle_db_schema_query(action_input):
    """
    성공: {"status":"ok", "db_schema": <any>}
    실패: {"status":"error", "reason": <str>}
    """
    try:
        db_schema_response = await asyncio.to_thread(get_db_schema)
        if db_schema_response.get("status") == "success":
            return {"status": "ok", "db_schema": db_schema_response.get("schema")}
        return {"status": "error", "reason": db_schema_response.get("message", "unknown error")}
    except Exception as e:
        return {"status": "error", "reason": f"db_schema_query failed: {e}"}


async def handle_get_specific_table_schema(action_input):
    """
    입력: {"table_names": [str, ...]}
    """
    try:
        table_names = action_input.get("table_names")
        if not table_names or not isinstance(table_names, list):
            return {"status": "error", "reason": "'table_names' must be a list"}

        db_schema_response = await asyncio.to_thread(get_db_schema_for_tables, table_names)
        if db_schema_response.get("status") == "success":
            return {"status": "ok", "db_schema": db_schema_response.get("schema")}
        return {"status": "error", "reason": db_schema_response.get("message", "unknown error")}
    except Exception as e:
        return {"status": "error", "reason": f"get_specific_table_schema failed: {e}"}


async def handle_execute_sql_query(action_input):
    """
    입력: {"sql_query": str, "is_write_operation": bool}
    """
    try:
        sql_query = action_input.get("sql_query")
        is_write_op = action_input.get("is_write_operation", False)
        if not sql_query:
            return {"status": "error", "reason": "'sql_query' is required"}

        sql_exec_result = await asyncio.to_thread(execute_raw_sql, sql_query, is_write_op)
        if sql_exec_result.get("status") == "success":
            return {"status": "ok", "sql_execution_result": sql_exec_result}
        return {"status": "error", "reason": sql_exec_result.get("message", "unknown error")}
    except Exception as e:
        return {"status": "error", "reason": f"execute_sql_query failed: {e}"}


# -------------------------
# 쉘 명령
# -------------------------
async def handle_execute_shell_command(action_input):
    """
    입력: {"command": str}
    성공: {"status":"ok","exit_code":0,"stdout": "...","stderr": "..."}
    실패: {"status":"error","reason": "...", "exit_code": <int>|None}
    """
    try:
        command = action_input.get("command")
        cmd = (command or "").strip()
        if not cmd:
            return {"status": "error", "reason": "'command' is required", "exit_code": None}

        # Windows 보호: cmd에서 './' 혹은 bash 전용 문법이 들어오면 bash 래핑
        if os.name == "nt":
            # 가장 흔한 문제: './script' 형태
            if cmd.startswith("./"):
                # git-bash 또는 WSL이 설치되어 있다면 아래 중 하나로 통일해 실행
                # 우선순위: WSL → git-bash → 그냥 그대로(최후)
                bash_cmds = []
                # WSL
                if shutil.which("wsl"):
                    bash_cmds.append(f'wsl bash -lc "{cmd}"')
                # git-bash (Git for Windows)
                git_bash = r"C:\Program Files\Git\bin\bash.exe"
                if os.path.exists(git_bash):
                    bash_cmds.append(f'"{git_bash}" -lc "{cmd}"')
                if bash_cmds:
                    cmd = bash_cmds[0]  # 가용한 첫 번째 방식 사용
                # 없으면 그대로 두되, 사용자가 ./ 대신 .\ 로 바꾸게끔 에러 메시지에서 안내하자
        if not command:
            return {"status": "error", "reason": "'command' is required", "exit_code": None}

        shell_result = await asyncio.to_thread(execute_shell, CommandRequest(command=cmd))
        if shell_result is None:
            return {"status": "error", "reason": "execute_shell returned None", "exit_code": None}

        exit_code = shell_result.get("exit_code")
        def _to_text(x):
            if isinstance(x, bytes):
                for enc in (locale.getpreferredencoding(False), "utf-8", "cp949", "euc-kr", "latin-1"):
                    try:
                        return x.decode(enc, errors="replace")
                    except Exception:
                        continue
                return x.decode("utf-8", errors="replace")
            return str(x)

        stdout_str = _to_text(shell_result.get("stdout", b"")).strip()
        stderr_str = _to_text(shell_result.get("stderr", b"")).strip()

        if exit_code == 0:
            return {"status": "ok", "exit_code": 0, "stdout": stdout_str, "stderr": stderr_str}
        else:
            return {
                "status": "error",
                "reason": "shell command failed",
                "exit_code": exit_code,
                "stdout": stdout_str,
                "stderr": stderr_str
            }
    except Exception as e:
        return {"status": "error", "reason": f"execute_shell_command failed: {e}", "exit_code": None}


# -------------------------
# 대화 검색/별칭/세션
# -------------------------
async def handle_search_conversation_history(action_input):
    """
    입력: {"session_id": str, "speaker_id": str|int|None, "keyword": str|None, "limit": int|None}
    성공: {"status":"ok", "search_result": <list|str>}
    실패: {"status":"error","reason": <str>}
    """
    try:
        session_id = action_input.get("session_id")
        if not session_id:
            return {"status": "error", "reason": "'session_id' is required"}

        speaker_id_str = action_input.get("speaker_id")
        speaker_id = None
        if speaker_id_str is not None:
            try:
                speaker_id = int(speaker_id_str)
            except Exception:
                # 화자 지정 없이 검색
                pass

        keyword = action_input.get("keyword")
        limit = action_input.get("limit", 30)

        query_str = (
            "SELECT u.user_name as speaker, c.message_content, c.timestamp "
            "FROM conversations c LEFT JOIN users u ON c.speaker_id = u.user_id "
        )
        where_clauses = ["c.session_id = :session_id"]
        params = {"session_id": session_id}

        if speaker_id is not None:
            where_clauses.append("c.speaker_id = :speaker_id")
            params["speaker_id"] = speaker_id

        if keyword:
            where_clauses.append("c.message_content LIKE :keyword")
            params["keyword"] = f"%{keyword}%"

        query_str += " WHERE " + " AND ".join(where_clauses)
        query_str += " ORDER BY c.timestamp DESC LIMIT :limit"
        params["limit"] = limit

        sql_result = await asyncio.to_thread(
            execute_raw_sql,
            sql_query=query_str,
            is_write_operation=False,
            params=params
        )

        if sql_result.get("status") == "success":
            data = sql_result.get("data") or []
            if not data:
                return {"status": "ok", "search_result": []}
            return {"status": "ok", "search_result": data}

        return {"status": "error", "reason": sql_result.get("message", "unknown error")}
    except Exception as e:
        return {"status": "error", "reason": f"search_conversation_history failed: {e}"}


async def handle_find_user_by_alias(action_input):
    try:
        alias_name = action_input.get("alias_name")
        if not alias_name:
            return {"status": "error", "reason": "'alias_name' is required"}

        query = (
            "SELECT u.user_id, u.user_name "
            "FROM users u JOIN aliases a ON u.user_id = a.user_id "
            "WHERE a.alias_name = :alias_name"
        )
        result = await asyncio.to_thread(
            execute_raw_sql,
            sql_query=query,
            params={"alias_name": alias_name}
        )
        if result.get("status") == "success" and result.get("data"):
            return {"status": "ok", "user": result["data"][0]}
        return {"status": "error", "reason": "alias not found"}
    except Exception as e:
        return {"status": "error", "reason": f"find_user_by_alias failed: {e}"}


async def handle_add_alias_to_user(action_input):
    try:
        user_id = action_input.get("user_id")
        alias_name = action_input.get("alias_name")
        if not user_id or not alias_name:
            return {"status": "error", "reason": "missing user_id/alias_name"}

        query = "INSERT INTO aliases (user_id, alias_name) VALUES (:user_id, :alias_name)"
        result = await asyncio.to_thread(
            execute_raw_sql,
            sql_query=query,
            params={"user_id": user_id, "alias_name": alias_name},
            is_write_operation=True
        )
        if result.get("status") == "success":
            return {"status": "ok", "rows_affected": result.get("rows_affected")}
        return {"status": "error", "reason": result.get("message", "unknown error")}
    except Exception as e:
        return {"status": "error", "reason": f"add_alias_to_user failed: {e}"}


async def handle_get_current_session_owner(action_input):
    try:
        session_id = action_input.get("session_id")
        if not session_id:
            return {"status": "error", "reason": "'session_id' is required"}

        query = (
            "SELECT u.user_id, u.user_name "
            "FROM users u JOIN sessions s ON u.user_id = s.owner_id "
            "WHERE s.session_id = :session_id"
        )
        result = await asyncio.to_thread(
            execute_raw_sql,
            sql_query=query,
            params={"session_id": session_id}
        )
        if result.get("status") == "success" and result.get("data"):
            return {"status": "ok", "owner": result["data"][0]}
        return {"status": "error", "reason": "owner not found"}
    except Exception as e:
        return {"status": "error", "reason": f"get_current_session_owner failed: {e}"}

def _resolve_project_path(p: str) -> Path:
    """
    상대 경로를 프로젝트 루트 기준으로 절대 경로로 변환한다.
    우선순위: 절대경로 그대로 -> (ROOT_DIR|CFG_ROOT_DIR)/p -> (CFG_BACKEND_DIR.parent)/p -> cwd/p
    """
    p_obj = Path(p)
    if p_obj.is_absolute():
        return p_obj

    candidates = []
    if ROOT_DIR:
        candidates.append(Path(ROOT_DIR) / p_obj)
    if CFG_ROOT_DIR:
        candidates.append(Path(CFG_ROOT_DIR) / p_obj)
    if CFG_BACKEND_DIR:
        # Backend 디렉토리의 부모(=프로젝트 루트) 기준으로도 시도
        candidates.append(Path(CFG_BACKEND_DIR).parent / p_obj)

    # 마지막 안전망: 현재 작업 디렉토리 기준
    candidates.append(Path.cwd() / p_obj)

    for c in candidates:
        try:
            # 심볼릭 링크/..\ 정규화
            c = c.resolve()
        except Exception:
            pass
        if c.exists():
            return c

    # 존재하지 않아도 가장 그럴듯한 후보를 반환 (CFG_ROOT_DIR 우선)
    if CFG_ROOT_DIR:
        return (Path(CFG_ROOT_DIR) / p_obj).resolve()
    if ROOT_DIR:
        return (Path(ROOT_DIR) / p_obj).resolve()
    return (Path.cwd() / p_obj).resolve()

async def handle_backup_file(action_input):
    """
    입력: {"file_path": "<원본 파일 경로>"}
    존재하지 않으면: {"status":"ok","skipped":true,"reason":"..."}
    존재하면 복사: {"status":"ok","src":"...","backup_path":"..."}
    """
    try:
        raw = action_input.get("file_path") or action_input.get("filepath")
        if not raw:
            return {"status": "error", "reason": "'file_path' is required"}

        src_path = _resolve_project_path(raw)

        if not src_path.exists():
            return {
                "status": "ok",
                "skipped": True,
                "reason": f"source not found (backup skipped)",
                "resolved": str(src_path)  # 디버깅용: 실제 확인한 절대경로
            }

        backup_dir = src_path.parent / "backup"
        backup_dir.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        dst_path = backup_dir / f"{src_path.stem}_{ts}{src_path.suffix}"
        shutil.copy2(src_path, dst_path)

        return {
            "status": "ok",
            "src": str(src_path),
            "backup_path": str(dst_path)
        }
    except Exception as e:
        return {"status": "error", "reason": f"backup_file failed: {e}"}
async def handle_append_file(file_path: str, content: str) -> dict:
    try:
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(content)
        return {"status": "ok", "file_path": str(p), "bytes_appended": len(content.encode("utf-8"))}
    except Exception as e:
        return {"status": "error", "reason": str(e), "file_path": file_path}
# -------------------------
# 등록 테이블
# -------------------------
basic_action_handlers = {
    "write_file": handle_write_file,
    "db_schema_query": handle_db_schema_query,
    "get_specific_table_schema": handle_get_specific_table_schema,
    "execute_sql_query": handle_execute_sql_query,
    "execute_shell_command": handle_execute_shell_command,
    "search_conversation_history": handle_search_conversation_history,
    "find_user_by_alias": handle_find_user_by_alias,
    "add_alias_to_user": handle_add_alias_to_user,
    "get_current_session_owner": handle_get_current_session_owner,
    "backup_file": handle_backup_file,
    "append_file": handle_append_file
}
