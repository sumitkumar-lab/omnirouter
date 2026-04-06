from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains.history_aware_retriever import create_history_aware_retriever
# from langchain.chains.retrieval import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.rag.vector_store import get_vector_store

def build_doc_assistant(api_key: str):
    """
    Constructs the conversational RAG pipeline.
    """
    # 1. Initialize our LLM (temperature=0 because we want factual answers, not creative ones)
    llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo", temperature=0)
    
    # 2. Connect to our Vector DB (k=2 means return the top 2 most relevant chunks)
    retriever = get_vector_store(api_key).as_retriever(search_kwargs={"k": 2})

    # ==========================================
    # STEP 1: The "Question Reformulation" Prompt
    # ==========================================
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"), # Injects our memory here
        ("human", "{input}"),
    ])
    
    # This chain automatically handles rewriting the query before searching
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # ==========================================
    # STEP 2: The "Final Answer" Prompt
    # ==========================================
    system_prompt = (
        "You are an elite AI Engineering Assistant. "
        "Use the following pieces of retrieved context to answer the question. "
        "If the answer is not contained in the context, say 'I don't know based on the documentation.' "
        "Do not make up an answer. Keep it concise.\n\n"
        "Context: {context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # This chain handles injecting the retrieved chunks into the {context} variable
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # ==========================================
    # STEP 3: Tie it all together
    # ==========================================
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain