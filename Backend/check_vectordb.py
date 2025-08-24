import chromadb
from pathlib import Path
import sys
import json

# gemini_router.py와 동일한 경로 설정
ROOT_DIR = Path("E:/Develop/ToyGemini")
BACKEND_DIR = ROOT_DIR / "Backend"
VECTOR_DB_PATH = str(BACKEND_DIR / 'vector_db')
COLLECTION_NAME = "kaede_tools"
# ChromaDB 클라이언트 연결
client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

def list_tools():
    """VectorDB에 저장된 모든 스킬을 나열합니다."""
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        all_data = collection.get(include=["metadatas"])

        if not all_data['ids']:
            print(f"✅ 컬렉션 '{COLLECTION_NAME}'에 스킬이 없습니다.")
            return

        print(f"✅ 컬렉션 '{COLLECTION_NAME}'에서 총 {len(all_data['ids'])}개의 스킬을 찾았습니다.\n")
        for i, tool_id in enumerate(all_data['ids']):
            print(f"--- 스킬 {i+1}: {tool_id} ---")
            metadata = all_data['metadatas'][i]
            print("  [Description]:", metadata.get('description'))
            print("-" * (len(tool_id) + 12) + "\n")

    except Exception as e:
        print(f"❌ VectorDB를 확인하는 중 오류가 발생했습니다: {e}")

def delete_tool(tool_id_to_delete: str):
    """지정된 ID의 스킬을 VectorDB에서 삭제합니다."""
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        
        existing_tool = collection.get(ids=[tool_id_to_delete])
        if not existing_tool['ids']:
            print(f"❌ 스킬 '{tool_id_to_delete}'을(를) VectorDB에서 찾을 수 없습니다.")
            return

        collection.delete(ids=[tool_id_to_delete])
        print(f"✅ 스킬 '{tool_id_to_delete}'이(가) VectorDB에서 성공적으로 삭제되었습니다.")

    except Exception as e:
        print(f"❌ 스킬을 삭제하는 중 오류가 발생했습니다: {e}")

def main():
    """스크립트 실행을 위한 메인 함수"""
    if len(sys.argv) == 3 and sys.argv[1].lower() == 'delete':
        tool_name = sys.argv[2]
        delete_tool(tool_name)
    elif len(sys.argv) == 1:
        list_tools()
        print("\nℹ️  특정 스킬을 삭제하려면 다음 명령어를 사용하세요:")
        print("   python check_vectordb.py delete <삭제할_스킬_이름>")
    else:
        print("❌ 잘못된 명령어입니다. 사용법: python check_vectordb.py [delete <스킬_이름>]")

if __name__ == "__main__":
    main()