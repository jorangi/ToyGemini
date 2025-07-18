from sqlalchemy import create_engine, Column, Integer, String, text, inspect
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import sessionmaker, declarative_base
import os
import dotenv
from datetime import datetime, date, time # date, time 추가
from decimal import Decimal # Decimal 추가
from uuid import UUID # UUID 추가
import base64 # bytes 인코딩을 위한 base64 추가

dotenv.load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:admin@localhost:3306/toygemini")
engine = create_engine(DATABASE_URL)

Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_db_tables():
    try:
        Base.metadata.create_all(engine)
        print("데이터베이스 테이블 생성 완료 또는 이미 존재함.")
    except Exception as e:
        print(f"데이터베이스 테이블 생성 실패: {e}")

def execute_raw_sql(sql_query: str, is_write_operation: bool = False):
    db_session = SessionLocal()
    try:
        result_proxy = db_session.execute(text(sql_query))
        if is_write_operation:
            db_session.commit()
            return {"status": "success", "message": "SQL 쿼리 실행 및 커밋 완료", "row_count": result_proxy.rowcount}
        else:
            columns = result_proxy.keys()
            rows = result_proxy.fetchall()
            
            # [최종 개선] 모든 잠재적 문제 데이터 타입을 JSON 친화적인 형태로 변환
            safe_results = []
            for row in rows:
                safe_row = {}
                for key, value in zip(columns, row):
                    if isinstance(value, (datetime, date, time)):
                        safe_row[key] = value.isoformat()
                    elif isinstance(value, Decimal):
                        safe_row[key] = str(value) # 정밀도를 유지하기 위해 문자열로 변환
                    elif isinstance(value, UUID):
                        safe_row[key] = str(value)
                    elif isinstance(value, bytes):
                        safe_row[key] = base64.b64encode(value).decode('utf-8') # Base64로 인코딩
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
