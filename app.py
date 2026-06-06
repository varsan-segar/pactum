import uuid
import streamlit as st
import extra_streamlit_components as stx
from datetime import datetime, timedelta

from langchain_core.messages import HumanMessage, AIMessage

from core.ingestion import ingestion_pipeline
from core.document_manager import list_documents, delete_document
from core.pipeline import rag_pipeline

from security.validators import UnsafeInputError, FileValidationError, validate_query, validate_file
from config import FILE_UPLOAD_PATH 

st.set_page_config(
    page_title="RAG Application",
    page_icon=":robot:",
    layout="wide"
)

cookie_manager = stx.CookieManager()

cookies = cookie_manager.get_all()

if len(cookies) == 0:
    st.stop()

if 'user_id' not in st.session_state:
    user_id = cookie_manager.get("user_id")

    if user_id:
        st.session_state.user_id = user_id
    else:
        user_id = str(uuid.uuid4())
        cookie_manager.set("user_id", user_id, expires_at=datetime.now() + timedelta(days=365))
        st.session_state.user_id = user_id
        
if 'history' not in st.session_state:
    st.session_state.history = []

if 'documents' not in st.session_state:
    st.session_state.documents = list_documents(user_id=st.session_state.user_id)

st.title("PDF RAG Q&A Application")

with st.sidebar:
    st.header("DOCUMENTS:")

    file = st.file_uploader(
        label="Upload PDF File",
        type=[".pdf"],
        max_upload_size=20
    )

    if file:
        try:
            is_new_file = True

            for document in st.session_state.documents:
                if document['file_name'] == file.name:
                    is_new_file = False
                    st.warning("This document is already indexed in the Vector DB.")

            if is_new_file:        
                USER_DIR = FILE_UPLOAD_PATH / st.session_state.user_id
                USER_DIR.mkdir(parents=True, exist_ok=True)

                FILE_PATH = USER_DIR / file.name

                with FILE_PATH.open("wb") as f:
                    f.write(file.getbuffer())

                file = validate_file(file=FILE_PATH)

                status = ingestion_pipeline(file=FILE_PATH, user_id=st.session_state.user_id)

                with st.expander(label="INGESTION STATUS:"):
                    for k, v in status.items():
                        st.write(f"{k} -> {v}")

                st.session_state.documents = list_documents(user_id=st.session_state.user_id)                    
        except FileNotFoundError as e:
            st.warning(e)
        except FileValidationError as e:
            st.warning(f"File Validation Error: {e}")
            FILE_PATH.unlink()
    
    if len(st.session_state.documents) > 0:
        with st.expander(label="DOCUMENT LIST:"):
            for document in st.session_state.documents:
                    st.write(f"{document['file_name']} -> {document['doc_id']}")
                    if st.button(label="Delete Document", key=f"delete_{document['doc_id']}"):
                        delete_document(user_id=st.session_state.user_id, doc_id=document['doc_id'])
                        st.session_state.documents = list_documents(user_id=st.session_state.user_id)
                        st.rerun()
    
    st.divider()

    st.write(f"""
    **USER ID:**
             
    `{st.session_state.user_id}`

    Save this ID. If cookies are cleared or you use a different browser/device, you'll need this ID to interact with your documents.
    """)

for msg in st.session_state.history:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

        if msg['role'] == "assistant":
            if msg["sources"]:
                with st.expander(label="SOURCES:"):
                    for i, chunk in enumerate(msg["sources"]):
                        st.write(f"{i}. [{chunk.file_name}] [{chunk.page_number}] [{chunk.chunk_index}]")
                        st.write(f"     {chunk.content}")

prompt = st.chat_input("Enter your query")

if prompt:
    try:
        query = validate_query(query=prompt)
        
        with st.chat_message("user"):
            st.markdown(query)

        st.session_state.history.append({"role": "user", "content": query})

        conversation_history = []

        for msg in st.session_state.history:
            if msg["role"] == "user":
                conversation_history.append(HumanMessage(msg["content"]))
            elif msg["role"] == "assistant":
                conversation_history.append(AIMessage(msg['content']))

        response, chunks = rag_pipeline(user_id=st.session_state.user_id, query=query, history=conversation_history[:-1])

        with st.chat_message("assistant"):
            full_response = st.write_stream(response)

            if chunks:
                with st.expander(label="SOURCES:"):
                    for i, chunk in enumerate(chunks):
                        st.write(f"{i}. [{chunk.file_name}] [{chunk.page_number}] [{chunk.chunk_index}]")
                        st.write(f"     {chunk.content}")

        st.session_state.history.append({"role": "assistant", "content": full_response, "sources": chunks})
    except UnsafeInputError as e:
        st.warning(f"Unsafe Input Error: {e}")