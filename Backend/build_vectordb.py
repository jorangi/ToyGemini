from sentence_transformers import SentenceTransformer
import chromadb
import json

# 1. 설정
MODEL_NAME = 'jhgan/ko-sroberta-multitask'
TOOL_DEFINITIONS_PATH = 'tools/definitions.json'
VECTOR_DB_PATH = 'vector_db'
COLLECTION_NAME = 'kaede_tools'

# 2. 모델 및 DB 클라이언트 초기화
print("임베딩 모델을 로드합니다...")
model = SentenceTransformer(MODEL_NAME)
client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

print(f"'{COLLECTION_NAME}' 컬렉션을 생성하거나 로드합니다...")
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# 3. 스킬 데이터 로드
with open(TOOL_DEFINITIONS_PATH, 'r', encoding='utf-8') as f:
    tools = json.load(f)

# 4. 벡터화 및 DB 저장
print(f"{len(tools)}개의 스킬을 Vector DB에 저장합니다...")
for tool in tools:
    description = tool['description']
    tool_name = tool['tool_name']

    # 설명을 벡터로 변환
    vector = model.encode(description).tolist()

    # Vector DB에 저장 (id, embedding, metadata)
    collection.add(
        ids=[tool_name],
        embeddings=[vector],
        metadatas=[tool] # 스킬 정의 전체를 메타데이터로 저장
    )

print("\n완료! Vector DB가 'vector_db' 폴더에 성공적으로 생성/업데이트되었습니다.")
print(f"현재 컬렉션에 {collection.count()}개의 스킬이 저장되어 있습니다.")