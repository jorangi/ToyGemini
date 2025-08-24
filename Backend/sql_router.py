from sqlalchemy import create_engine, Column, Integer, String, Text, inspect, DateTime, func, text, ForeignKey
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
import os
import dotenv
from datetime import datetime, date, time # date, time 추가
from decimal import Decimal # Decimal 추가
from uuid import UUID # UUID 추가
import base64 # bytes 인코딩을 위한 base64 추가
from collections import defaultdict, deque
from sqlalchemy.dialects.mysql import JSON

# Base는 이미 위에서 선언됨
recent_messages = defaultdict(lambda: deque(maxlen=30))

dotenv.load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:admin@localhost:3306/toygemini")
engine = create_engine(DATABASE_URL)

Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class User(Base):
    __tablename__ = "users"
    user_id = Column(Integer, primary_key=True, autoincrement=True)
    user_name = Column(String(255), nullable=False)
    aliases = relationship("Alias", back_populates="user")
    sessions = relationship("Session", back_populates="owner")

class Alias(Base):
    __tablename__ = "aliases"
    alias_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    alias_name = Column(String(255), nullable=False, unique=True)
    user = relationship("User", back_populates="aliases")

class Session(Base):
    __tablename__ = "sessions"
    session_id = Column(String(255), primary_key=True)
    owner_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    owner = relationship("User", back_populates="sessions")

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), index=True)
    timestamp = Column(DateTime, default=func.now())
    speaker_id = Column(Integer, ForeignKey("users.user_id"), nullable=True)
    message_content = Column(Text)

# 세션별 최근 30개 대화 메시지 캐시

class ToolCallLog(Base):
    __tablename__ = 'tool_call_logs'
    log_id = Column(Integer, primary_key=True, autoincrement=True) # 기존 'log_id' 컬럼에 매핑
    session_id = Column(String(255), nullable=False, index=True)
    request_id = Column(String(255), nullable=False)
    iteration = Column(Integer, nullable=False) # 기존 'iteration' 컬럼 추가
    tool_name = Column(String(255), nullable=False)
    parameters = Column(JSON, nullable=True) # 기존 'parameters' JSON 컬럼에 매핑
    timestamp = Column(DateTime, default=func.now()) # 기존 'timestamp' 컬럼에 매핑
    # tool_input, tool_output, user_goal, workflow_pattern은 기존 스키마에 없으므로 제거
    # 만약 이 데이터가 필요하다면 'parameters' JSON 내부의 필드로 포함시켜야 합니다.

def create_db_tables():
    try:
        Base.metadata.create_all(engine)
        print("데이터베이스 테이블 생성 완료 또는 이미 존재함.")
    except Exception as e:
        print(f"데이터베이스 테이블 생성 실패: {e}")

def execute_raw_sql(sql_query: str, is_write_operation: bool = False, params: dict = None):
    """
    SQL 쿼리를 안전하게 실행합니다.
    - sql_query: 실행할 SQL 문자열
    - is_write_operation: INSERT, UPDATE, DELETE 등 쓰기 작업 여부
    - params: SQL 파라미터 바인딩을 위한 딕셔너리
    """
    db_session = SessionLocal()
    try:
        result_proxy = db_session.execute(text(sql_query), params)
        
        if is_write_operation:
            db_session.commit()
            return {"status": "success", "message": "SQL 쿼리 실행 및 커밋 완료", "row_count": result_proxy.rowcount}
        else:
            columns = result_proxy.keys()
            rows = result_proxy.fetchall()
            
            safe_results = []
            for row in rows:
                safe_row = {}
                for key, value in zip(columns, row):
                    if isinstance(value, (datetime, date, time)):
                        safe_row[key] = value.isoformat()
                    elif isinstance(value, Decimal):
                        safe_row[key] = str(value)
                    elif isinstance(value, UUID):
                        safe_row[key] = str(value)
                    elif isinstance(value, bytes):
                        safe_row[key] = base64.b64encode(value).decode('utf-8')
                    else:
                        safe_row[key] = value
                safe_results.append(safe_row)

            return {"status": "success", "data": safe_results, "message": "SQL 쿼리 실행 완료"}
            
    except SQLAlchemyError as e:
        db_session.rollback()
        return {"status": "error", "message": f"SQL 쿼리 실행 중 오류 발생: {str(e)}"}
    except Exception as e:
        db_session.rollback()
        return {"status": "error", "message": f"예상치 못한 오류 발생: {str(e)}"}
    finally:
        db_session.close()

# [✨ 최종 수정] DB 스키마 조회 함수
def get_db_schema():
    """
    현재 데이터베이스의 모든 테이블 이름과 각 테이블의 컬럼 정보를 조회하여 반환합니다.
    """
    inspector = inspect(engine)
    schema_info = {}
    try:
        table_names = inspector.get_table_names()
        for table_name in table_names:
            columns = inspector.get_columns(table_name)
            column_info = []
            for col in columns:
                column_info.append({
                    "name": col['name'],
                    "type": str(col['type']),
                    "nullable": col['nullable'],
                    # [수정] 딕셔너리 키 접근 문법 오류를 .get() 메서드로 수정
                    "primary_key": col.get('primary_key', False)
                })
            schema_info[table_name] = column_info
        return {"status": "success", "schema": schema_info, "message": "DB 스키마 조회 완료"}
    except Exception as e:
        print(f"DB 스키마 조회 실패: {e}")
        return {"status": "error", "message": f"DB 스키마 조회 중 오류 발생: {str(e)}"}
        

def get_db_schema_for_tables(table_names: list):
    """
    주어진 테이블 이름 목록에 대해서만 스키마 정보를 조회하여 반환합니다.
    """
    # engine 객체는 이 파일의 전역 범위에 이미 있으므로, 바로 사용하면 됩니다.
    inspector = inspect(engine)
    schema_info = {}
    try:
        # 인자로 받은 테이블 목록을 순회합니다.
        for table_name in table_names:
            # 테이블이 실제로 존재하는지 확인 후 진행
            if inspector.has_table(table_name):
                columns = inspector.get_columns(table_name)
                column_info = []
                for col in columns:
                    column_info.append({
                        "name": col['name'],
                        "type": str(col['type']),
                        "nullable": col['nullable'],
                        "primary_key": col.get('primary_key', False)
                    })
                schema_info[table_name] = column_info
        return {"status": "success", "schema": schema_info, "message": "특정 테이블 스키마 조회 완료"}
    except Exception as e:
        print(f"특정 테이블 스키마 조회 실패: {e}")
        return {"status": "error", "message": f"특정 테이블 스키마 조회 중 오류 발생: {str(e)}"}
def save_message(session_id: str, message_content: str, db_session, speaker_id: int | None = None):
    """
    주어진 대화 내용을 DB에 저장합니다.
    speaker_id가 제공되면 사용자의 발언으로, None이면 카에데의 발언으로 기록합니다.
    """
    try:
        new_conversation = Conversation(
            session_id=session_id,
            speaker_id=speaker_id,
            message_content=message_content,
            timestamp=datetime.now(),
        )
        db_session.add(new_conversation)
        db_session.commit()
        print(f"[📝 Memory Saved] Session: {session_id}, Speaker ID: {speaker_id}")
    except Exception as e:
        db_session.rollback()
        # 오류 발생 시 더 자세한 로그를 남깁니다.
        print(f"[❌ Memory Error] 메시지 DB 저장 실패: {e}")
        import traceback
        traceback.print_exc()
