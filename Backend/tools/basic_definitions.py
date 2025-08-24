from google.genai import types

basic_tool_definitions = [
    types.FunctionDeclaration(
        name="search_conversation_history",
        description="과거 대화 기록을 검색하여 이전 대화의 맥락을 파악합니다.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "session_id": types.Schema(type=types.Type.STRING, description="검색할 대화의 현재 세션 ID."),
                "speaker_id": types.Schema(type=types.Type.STRING, description="검색할 발언자 이름."),
                "keyword": types.Schema(type=types.Type.STRING, description="검색할 특정 키워드."),
                "limit": types.Schema(type=types.Type.INTEGER, description="가져올 최신 대화의 최대 개수(기본 10)."),
            },
            required=["session_id"]
        )
    ),
    types.FunctionDeclaration(
        name="initiateToolCreationProcess",
        description="[내부 관리] 성공적으로 완료된 작업 흐름을 분석하여 반복 패턴일 경우 새로운 통합 도구 생성을 시작합니다.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "completed_workflow": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
                "user_goal": types.Schema(type=types.Type.STRING),
            },
            required=["completed_workflow", "user_goal"]
        )
    ),
    types.FunctionDeclaration(
        name="generatePythonCodeForTool",
        description="[내부 관리] 도구 명세와 종속성 그래프를 바탕으로 실행 코드 생성.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "tool_definition": types.Schema(type=types.Type.STRING),
                "dependency_graph": types.Schema(type=types.Type.STRING),
            },
            required=["tool_definition", "dependency_graph"]
        )
    ),
    types.FunctionDeclaration(
        name="registerNewTool",
        description="[내부 관리] 생성된 도구를 파일/VectorDB에 등록.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "tool_definition_json": types.Schema(type=types.Type.STRING),
                "python_code_string": types.Schema(type=types.Type.STRING),
            },
            required=["tool_definition_json", "python_code_string"]
        )
    ),

    # DB 계열들: 메타키 제거
    types.FunctionDeclaration(
        name="db_schema_query",
        description="현재 데이터베이스의 모든 테이블 이름과 각 테이블의 스키마를 조회합니다."
    ),
    types.FunctionDeclaration(
        name="get_specific_table_schema",
        description="지정된 테이블들의 스키마 정보만 반환합니다.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "table_names": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING),
                    description="스키마를 조회할 테이블 이름 리스트."
                )
            },
            required=["table_names"]
        )
    ),
    types.FunctionDeclaration(
        name="execute_sql_query",
        description="주어진 SQL 쿼리를 실행합니다.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "sql_query": types.Schema(type=types.Type.STRING, description="실행할 전체 SQL 쿼리"),
                "is_write_operation": types.Schema(type=types.Type.BOOLEAN, description="데이터 변경 여부"),
            },
            required=["sql_query", "is_write_operation"]
        )
    ),

    # 파일 쓰기: 키 이름을 'file_path' 또는 'filepath' 중 하나로 표준화 (아래는 file_path 기준)
    types.FunctionDeclaration(
        name="write_file",
        description="파일 경로와 내용을 받아 생성/덮어쓰기 합니다.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "file_path": types.Schema(type=types.Type.STRING, description="생성/덮어쓸 파일 경로"),
                "content": types.Schema(type=types.Type.STRING, description="파일에 쓸 전체 내용"),
            },
            required=["file_path", "content"]
        )
    ),

    # 쉘 실행: 유지할 거면 메타 제거 + 별도 게이트는 런타임에서
    types.FunctionDeclaration(
        name="execute_shell_command",
        description="운영체제 쉘(CMD) 명령을 실행합니다. (주의: 제한/감사 대상)",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "command": types.Schema(type=types.Type.STRING, description="실행할 CMD 명령어"),
            },
            required=["command"]
        )
    ),

    types.FunctionDeclaration(
        name="final_response",
        description="사용자에게 최종 답변을 합니다.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "answer": types.Schema(type=types.Type.STRING, description="사용자에게 전달할 최종 답변"),
            },
            required=["answer"]
        )
    ),
    types.FunctionDeclaration(
        name="backup_file",
        description="지정한 파일을 동일 폴더의 'backup' 디렉토리에 타임스탬프 파일명으로 복사합니다.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "file_path": types.Schema(
                    type=types.Type.STRING,
                    description="백업할 원본 파일 경로(예: Frontend/public/longText.txt)"
                )
            },
            required=["file_path"]
        )
    ),
    types.FunctionDeclaration(
        name="append_file",
        description="Append text to a file (creates if not exists).",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "file_path": types.Schema(type=types.Type.STRING, description="Target file path"),
                "content": types.Schema(type=types.Type.STRING, description="Text chunk to append"),
            },
            required=["file_path", "content"]
        )
    )
]