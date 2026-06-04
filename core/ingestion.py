import uuid
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import ACTIVE_EMBEDDING_PROVIDER, CHUNK_SIZE, CHUNK_OVERLAP, VECTOR_DB_PATH, VECTOR_DB_COLLECTION_NAME, get_embedding_model

def load_document(file: Path, user_id: str, doc_id: str):
    loader = PyPDFLoader(file)
    documents = loader.load()

    file_name = file.name
    for doc in documents:
        doc.metadata.update({
            "user_id": user_id,
            "doc_id": doc_id,
            "file_name": file_name
        })
    
    return documents

def split_document(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )

    chunks = splitter.split_documents(documents=docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "chunk_index": i + 1
        })

    return chunks

def embedding_function():
    if ACTIVE_EMBEDDING_PROVIDER.upper() == "OPENAI":
        embedding = OpenAIEmbeddings(**get_embedding_model())

        return embedding
    
    embedding = HuggingFaceEmbeddings(**get_embedding_model())

    return embedding

def store_document(chunks, embedding):
    vector_store = Chroma(
        collection_name=VECTOR_DB_COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=VECTOR_DB_PATH
    )

    vector_store.add_documents(chunks)

def ingestion_pipeline(file: Path, user_id: str):
    doc_id = str(uuid.uuid4())

    docs = load_document(file=file, user_id=user_id, doc_id=doc_id)
    chunks = split_document(docs=docs)
    embedding = embedding_function()
    store_document(chunks=chunks, embedding=embedding)

    return {"doc_id": doc_id, "chunks_length": len(chunks), "status": "success"}