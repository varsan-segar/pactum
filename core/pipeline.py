from core.retrieval import retrieval_pipeline
from core.conversation import llm_pipeline

def rag_pipeline(user_id, query, history):
    chunks = retrieval_pipeline(user_id=user_id, query=query)

    if not chunks:
        return iter(["No relevant context was found for your query."]), chunks

    response = llm_pipeline(query=query, context=chunks, history=history)

    return response, chunks