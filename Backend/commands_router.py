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

def sanitize_echo(cmd: str) -> str:
    # echo로 시작하는 줄에서, > 전까지의 문자열이 특수문자(#, <, > 등)를 포함하면 따옴표 감싸기
    m = re.match(r'(echo )(.+?)( >{1,2} .+)', cmd)
    if m:
        content = m.group(2).strip()
        if any(x in content for x in '#<>'):
            content = f'"{content}"'
        return m.group(1) + content + m.group(3)
    return cmd

# commands_router.py 파일의 execute_shell 함수를 아래 코드로 교체하세요.
import re # 파일 상단에 re 모듈을 import 해주세요.

@router.post("/execute-shell")
def execute_shell(req: CommandRequest):
    # 1. 평소에는 실행을 금지할 명령어 목록은 그대로 둡니다.
    blocked = [
        'del', 'rm', 'format', 'shutdown', 'regedit', 'erase', 'rmdir',
        'wsl', 'start', 'curl', 'ftp', 'runas', 'schtasks',
        'net user', 'net localgroup', 'vssadmin', 'diskpart', 'bcdedit'
    ]

    command_to_run = req.command.strip()
    is_forced = False

    # 2. 'forcecommand(...)' 패턴이 있는지 확인합니다.
    force_match = re.match(r'^\s*forcecommand\((.*)\)\s*$', command_to_run, re.DOTALL)
    
    if force_match:
        # 패턴이 일치하면, 안쪽의 실제 명령어만 추출하고 강제 실행 플래그를 True로 설정합니다.
        command_to_run = force_match.group(1).strip()
        is_forced = True
        print(f"🛡️ 강제 실행 모드 감지. 실행할 명령어: {command_to_run}")
    
    # 3. 강제 실행 모드가 아닐 경우에만 금지된 명령어를 확인합니다.
    if not is_forced:
        lowered_cmd = command_to_run.lower()
        for bad in blocked:
            if re.search(rf'\b{re.escape(bad)}\b', lowered_cmd):
                # 금지된 명령어일 경우, 400 오류와 함께 강제 실행 방법을 안내합니다.
                raise HTTPException(
                    status_code=400,
                    detail=f"금지된 명령어입니다: '{bad}'. 강제 실행을 원하시면 'forcecommand({command_to_run})' 형식으로 요청하세요."
                )

    # 4. 최종적으로 실행할 명령어를 처리합니다. (이전과 동일)
    try:
        raw_commands = command_to_run.split('\n')
        commands = []
        for line in raw_commands:
            for part in line.split('&'):
                cmd = part.strip()
                if cmd:
                    commands.append(sanitize_echo(cmd))

        full_command = " & ".join(commands)
        
        print("실행:", full_command)
        result = subprocess.run(
            f'chcp 65001 > nul && {full_command}',
            capture_output=True,
            text=True,
            shell=True,
            timeout=30,
            cwd=ROOT_DIR,
            encoding='utf-8',
            errors='replace'
        )
        
        stdout = result.stdout.strip()
        if "Active code page: 65001" in stdout:
            stdout = stdout.replace("Active code page: 65001", "").strip()
            
        return {
            "stdout": stdout,
            "stderr": result.stderr.strip(),
            "exit_code": result.returncode
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="명령 실행 시간 초과")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"명령 실행 오류: {str(e)}")
