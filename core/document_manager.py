from langchain_chroma import Chroma
from config import VECTOR_DB_PATH, VECTOR_DB_COLLECTION_NAME

def list_documents(user_id: str):
    vector_store = Chroma(
        collection_name=VECTOR_DB_COLLECTION_NAME,
        persist_directory=VECTOR_DB_PATH
    )

    documents = vector_store.get(
        where={"user_id": user_id},
        include=["metadatas"]
    )

    docs_list = {}

    for doc in documents['metadatas']:
        docs_list[doc['doc_id']] = doc['file_name']
    
    if docs_list:
        return [{"doc_id": doc_id, "file_name": file_name} for doc_id, file_name in docs_list.items()]
    
    return []

def delete_document(user_id: str, doc_id: str):
    vector_store = Chroma(
        collection_name=VECTOR_DB_COLLECTION_NAME,
        persist_directory=VECTOR_DB_PATH
    )

    documents = vector_store.get(
        where={"user_id": user_id, "doc_id": doc_id}
    )

    ids = documents.get("ids")

    if not ids:
        return {"status": "error", "chunks_length": 0}
    
    vector_store.delete(ids=ids)
    
    return {"status": "success", "chunks_length": len(ids)}