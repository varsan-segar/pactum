from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import ACTIVE_EMBEDDING_PROVIDER, TOP_K, VECTOR_DB_PATH, VECTOR_DB_COLLECTION_NAME, get_embedding_model

class Chunk(BaseModel):
    content: str
    chunk_index: int
    page_number: int
    file_name: str
    doc_id: str

def embedding_function():
    if ACTIVE_EMBEDDING_PROVIDER.upper() == "OPENAI":
        embedding = OpenAIEmbeddings(**get_embedding_model())

        return embedding
    
    embedding = HuggingFaceEmbeddings(**get_embedding_model())

    return embedding

def retrieve_chunks(user_id, query, embedding):
    vector_store = Chroma(
        collection_name=VECTOR_DB_COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=VECTOR_DB_PATH
    )

    return vector_store.similarity_search(
        query=query,
        k=TOP_K,
        filter={"user_id": user_id}
    )

def format_chunks(chunks):
    formatted_chunks = []
    
    for chunk in chunks:
        formatted_chunks.append(Chunk(content=chunk.page_content,
                                      chunk_index=chunk.metadata["chunk_index"],
                                      page_number=chunk.metadata['page'] + 1,
                                      file_name=chunk.metadata['file_name'],
                                      doc_id=chunk.metadata['doc_id']))

    return formatted_chunks

def retrieval_pipeline(user_id, query):
    embedding = embedding_function()

    chunks = retrieve_chunks(user_id=user_id, query=query, embedding=embedding)
    formatted_chunks = format_chunks(chunks=chunks)

    return formatted_chunks