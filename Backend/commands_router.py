from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import subprocess
from pathlib import Path
import shlex
import re

router = APIRouter()

ROOT_DIR = Path("E:/Develop/ToyGemini")

class WriteFileRequest(BaseModel):
    filepath: str
    content: str

@router.post("/write-file")
def write_file(req: WriteFileRequest):
    abs_path = (ROOT_DIR / req.filepath).resolve()

    # 루트 밖 접근 차단
    if not str(abs_path).startswith(str(ROOT_DIR)):
        raise HTTPException(status_code=403, detail="허용되지 않은 경로입니다")

    try:
        abs_path.parent.mkdir(parents=True, exist_ok=True)  # 폴더 없으면 생성
        with open(abs_path, "w", encoding="utf-8") as f: # 파일 저장 시에도 UTF-8 명시
            f.write(req.content)
        return {"status": "success", "path": str(abs_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 저장 실패: {str(e)}")

class CommandRequest(BaseModel):
    command: str

def resolve_command_path(raw_cmd: str) -> str:
    # 복합 명령(cmd /c ...)은 손대지 않고 바로 반환
    if raw_cmd.strip().lower().startswith("cmd /c"):
        return raw_cmd

    # 단일 명령어만 경로 검증 및 치환 (ex: type "./test.txt")
    try:
        tokens = shlex.split(raw_cmd, posix=False)
    except Exception:
        tokens = raw_cmd.strip().split()

    for i, token in enumerate(tokens):
        # CMD 예약어, for/do/if 등은 무시
        if token.lower() in ['cmd', '/c', 'cmd.exe', 'for', 'do', 'if', 'else', 'in', 'findstr']:
            continue
        # 배치 변수(%f, %%f) 등은 무시
        if '%' in token:
            continue
        # "명령" 뒤 따르는 실제 경로만 엄격히 치환
        unquoted = token.strip('"').strip("'")
        if (
            (unquoted.startswith('./') or unquoted.startswith('.\\'))
            and ('.' in unquoted or '/' in unquoted or '\\' in unquoted) # 경로 문자 포함 확인
        ):
            # 절대 경로로 변환 및 ROOT_DIR 제한 검사
            abs_path = (ROOT_DIR / unquoted).resolve()
            if not str(abs_path).startswith(str(ROOT_DIR)):
                raise HTTPException(status_code=403, detail="허용되지 않은 경로입니다")
            tokens[i] = f'"{str(abs_path)}"' # 변환된 절대 경로를 다시 따옴표로 감쌈
    return ' '.join(tokens)


@router.post("/execute-shell")
def execute_shell(req: CommandRequest):
    blocked = [
        'del', 'rm', 'format', 'shutdown', 'regedit', 'erase', 'rmdir',
        'powershell', 'wsl', 'start', 'curl', 'ftp', 'runas', 'schtasks',
        'net user', 'net localgroup', 'vssadmin', 'diskpart', 'bcdedit'
    ]

    lowered = req.command.lower()
    print(f"Checking blocked words in command: {lowered}")

    for bad in blocked:
        if re.search(rf'\b{re.escape(bad)}\b', lowered):
            raise HTTPException(status_code=400, detail=f"금지된 명령어입니다: {bad}")

    try:
        # 경로 변환된 안전한 명령어를 가져옴
        safe_cmd = resolve_command_path(req.command)
        # 문자열 이스케이프 관련 문제 발생 가능성 제거 (필요시)
        safe_cmd = safe_cmd.replace('\\"', '"').replace('\\\\', '\\')
        
        print("Executing command:", safe_cmd)

        # ✨ 핵심 변경 사항: chcp 65001 추가 및 encoding='utf-8' 지정
        # 'chcp 65001 > nul'로 코드 페이지를 UTF-8로 변경하고,
        # 이어서 원래 명령어를 실행합니다. '&&'는 앞선 명령이 성공해야 다음을 실행합니다.
        full_cmd_with_encoding = f"chcp 65001 > nul && {safe_cmd}"

        result = subprocess.run(
            full_cmd_with_encoding, # 변경된 전체 명령어
            capture_output=True,
            text=True,
            shell=True,
            timeout=5,
            cwd=ROOT_DIR,
            encoding='utf-8', # ✨ subprocess가 UTF-8로 입출력을 처리하도록 명시
            errors='replace' # 디코딩 오류 시 대체 문자로 처리
        )
        
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        # chcp 명령의 성공 메시지가 stdout에 포함될 수 있으므로 제거
        if "Active code page: 65001" in stdout:
            stdout = stdout.replace("Active code page: 65001", "").strip()

        return {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": result.returncode
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="명령 실행 시간 초과")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"명령 실행 오류: {str(e)}")