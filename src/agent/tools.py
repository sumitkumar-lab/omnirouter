"""
Let's officially give your agent its brain.

We are going to use LangChain's @tool decorator. 
This magical little wrapper takes your standard Python function, 
reads the type hints (like query: str), reads the docstring, and automatically 
translates the entire thing into a strict JSON schema that OpenAI and Anthropic natively understand .
"""
import os
from langchain_core.tools import tool
from src.rag.vector_store import get_vector_store
# web search package...
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool
import requests
import sqlite3

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

python_repl = PythonREPLTool()


# ==========================================
# 1. THE CRAG GRADER BRAIN
# ==========================================
class DocumentGrader(BaseModel):
    is_relevant: str = Field(description="Return 'yes' if the document is relevant to the query, 'no' if not.")

# We use temperature=0 for strict, uncreative grading
grader_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
structured_grader = grader_llm.with_structured_output(DocumentGrader)

grader_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an automated CRAG Grader. Your job is to assess if a retrieved document contains information relevant to the user's query. If it contains ANY helpful keywords or concepts, score 'yes'. If it is completely off-topic, score 'no'. Do not guess."),
    ("human", "Query: {query}\n\nDocument: {document}")
])

grader_chain = grader_prompt | structured_grader


# ==========================================
# 2. THE SELF-REFLECTIVE RAG TOOL
# ==========================================
@tool
def search_documentation(query: str) -> str:
    """
    Searches the internal engineering documentation for information about the OmniRouter, 
    LangChain, LangGraph, or general AI engineering concepts.
    """
    print(f"\n--- [TOOL EXECUTION] Searching Vector DB for: '{query}' ---")
    try:
        db = get_vector_store()
        # Step 1: High Recall (Pull top 3 documents)
        results = db.similarity_search(query, k=3)
        
        if not results:
            return "No documents found in the database at all."

        # Step 2: The CRAG Reflection Loop
        print("🔍 [CRAG] Grading retrieved documents for relevance...")
        valid_docs = []
        
        for i, doc in enumerate(results):
            score = grader_chain.invoke({"query": query, "document": doc.page_content})
            if score.is_relevant == 'yes':
                print(f"   ✅ Doc {i+1}: Relevant")
                valid_docs.append(doc.page_content)
            else:
                print(f"   ❌ Doc {i+1}: Garbage (Discarding)")

        # Step 3: The Autonomous Pivot
        # If all documents were garbage, we force the Supervisor to use the Web Search tool!
        if not valid_docs:
            print("⚠️ [CRAG] All local documents rejected! Instructing Supervisor to use Web Search.")
            return (
                "INTERNAL SYSTEM DIRECTIVE: The local database does not contain the answer to this query. "
                "The documents retrieved were irrelevant. You MUST immediately use the 'search_web' tool "
                "to find this information on the live internet. Do not apologize, just run the web tool."
            )

        # If we have valid docs, return only the good ones
        return "\n\n".join(valid_docs)
        
    except Exception as e:
        print(f"\n🚨 [TOOL CRASHED]: {str(e)}") 
        return f"Error executing search: {str(e)}"
    

# The @tool decorator converts this Python function into an LLM-readable JSON schema
# Internal database tools
@tool
def search_documentation(query: str) -> str:
    """
    Searches the internal engineering documentation for information about the OmniRouter, 
    LangChain, LangGraph, or general AI engineering concepts.
    
    Use this tool WHENEVER the user asks a technical question about how the system works, 
    fallback protocols, or specific coding architecture. Do NOT use this for general greetings.
    
    Args:
        query: The specific search term to look up in the database. 
               It should be a standalone, highly descriptive phrase.
    """
    print(f"\n--- [TOOL EXECUTION] Searching Vector DB for: '{query}' ---")
    
    # In production, ensure your API key is loaded securely
    api_key = os.getenv("HF_TOKEN") 
    
    try:
        db = get_vector_store(api_key)
        # Retrieve the top 2 most relevant chunks
        results = db.similarity_search(query, k=2)
        if not results:
            return "No relevant information found in the documentation." 
        # We must return a STRING, not a list of objects, so the LLM can read it easily
        combined_text = "\n\n".join([doc.page_content for doc in results])
        return combined_text
        
    except Exception as e:
        # FIXED: Print the error to the server terminal so we can see it!
        print(f"\n🚨 [TOOL CRASHED]: {str(e)}")
        return f"Error executing search: {str(e)}"
    
    
# 2. NEW: The Global Researcher Tool
@tool
def search_web(query: str) -> str:
    """
    Searches the live internet for up-to-date information, news, or concepts 
    that are NOT found in the internal documentation. Use this for current events.
    """
    print(f"\n--- [TOOL EXECUTION] Searching the Live Web for: '{query}' ---")
    search = DuckDuckGoSearchRun()
    try:
        # This executes a live DuckDuckGo search and returns a text summary
        return search.invoke(query)
    except Exception as e:
        print(f"\n🚨 [WEB TOOL CRASHED]: {str(e)}")
        return f"Error executing web search: {str(e)}"

# 3. NEW: The Data Scientist Tool
@tool
def execute_python_code(code: str) -> str:
    """
    Executes Python code in a REPL environment and returns the terminal output.
    Use this tool to solve complex math problems, deterministic logic, or data analysis.
    
    CRITICAL RULE: You MUST use print() statements at the end of your script to output the final answer, 
    otherwise you will not see the result.
    
    Args:
        code: A valid, raw Python script string. Do NOT wrap it in markdown block quotes.
    """
    print(f"\n--- [TOOL EXECUTION] Running Python Script ---")
    print(f"Code Payload:\n{code}\n")
    try:
        # Execute the string of code and return what prints to the console
        result = python_repl.invoke(code)
        return f"Terminal Output:\n{result}"
    except Exception as e:
        print(f"\n🚨 [PYTHON TOOL CRASHED]: {str(e)}")
        return f"Error executing Python: {str(e)}"

# 4.
@tool
def get_github_issue(repo: str, issue_number: int) -> str:
    """
    Fetches the details of a specific GitHub issue from a public repository.
    Use this when the user asks to summarize or read a bug report.
    
    CRITICAL RULE: The returned description WILL be truncated to save memory. 
    You MUST NOT complain that it is truncated. You MUST NOT try to fetch it again.
    Simply provide the best summary possible using only the text you are given.
    
    Args:
        repo: The exact repository name in "owner/repo" format.
        issue_number: The integer issue number.
    """
    print(f"\n--- [TOOL EXECUTION] Fetching GitHub Issue #{issue_number} from {repo} ---")
    
    url = f"https://api.github.com/repos/{repo}/issues/{issue_number}"
    
    try:
        # We hit the public GitHub API (No auth required for public repos)
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            title = data.get("title", "No Title")
            body = data.get("body", "No Body")
            
            # Combine the data and strictly truncate to protect Llama 3's brain!
            full_text = f"ISSUE TITLE: {title}\n\nISSUE DESCRIPTION:\n{body}"
            return full_text[:2000] # + "\n... [Truncated]"
        else:
            return f"Failed to fetch issue. GitHub API status code: {response.status_code}"
            
    except Exception as e:
        print(f"\n🚨 [GITHUB TOOL CRASHED]: {str(e)}")
        return f"Error executing GitHub API call: {str(e)}"
    

# ... (keep existing tools) ...

@tool
def execute_sql_query(query: str) -> str:
    """
    Executes a SQL SELECT query against the corporate SQLite database.
    Use this to answer questions about employee revenue, departments, or sales metrics.
    
    CRITICAL RULE: You must write valid SQLite syntax.
    
    DATABASE SCHEMA:
    Table Name: sales
    Columns: 
      - id (INTEGER PRIMARY KEY)
      - employee_name (TEXT)
      - department (TEXT)
      - revenue (REAL)
      - month (TEXT)
      
    Args:
        query: A valid SQL SELECT statement.
    """
    print(f"\n--- [TOOL EXECUTION] Running SQL Query: {query} ---")
    
    # Security: In a real system, use a strictly read-only user role here!
    if not query.strip().upper().startswith("SELECT"):
        return "Error: For security reasons, only SELECT queries are allowed."
        
    try:
        conn = sqlite3.connect("company.db")
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return "Query executed successfully, but returned 0 rows."
            
        # Format the rows nicely for the LLM to read
        formatted_results = "\n".join([str(row) for row in rows])
        return f"Query Results:\n{formatted_results}"
        
    except sqlite3.Error as e:
        print(f"\n🚨 [SQL TOOL CRASHED]: {str(e)}")
        return f"SQL Execution Error: {str(e)}"
    
@tool
def search_documentation(query: str) -> str:
    """
    Searches the internal engineering documentation for information about the OmniRouter.
    """
    print(f"\n--- [TOOL EXECUTION] Searching Vector DB for: '{query}' ---")
    try:
        db = get_vector_store()
        
        # ==========================================
        # STAGE 1 & 2: THE RERANKING PIPELINE
        # ==========================================
        print("🗄️ [STAGE 1] Pulling top 10 documents for high recall...")
        base_retriever = db.as_retriever(search_kwargs={"k": 10})
        
        print("🧠 [STAGE 2] Cross-Encoder Reranking the results...")
        # We use a tiny, lightning-fast cross-encoder model from HuggingFace
        cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        # We tell the reranker to only keep the absolute top 3 highest-scoring documents
        reranker = CrossEncoderReranker(model=cross_encoder, top_n=3)
        
        # We combine them into a single pipeline
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker, 
            base_retriever=base_retriever
        )
        
        # Execute the two-stage search!
        results = compression_retriever.invoke(query)
        
        if not results:
            return "No documents found in the database at all."

        # ==========================================
        # STAGE 3: THE CRAG REFLECTION LOOP (Keep your existing code!)
        # ==========================================
        print("🔍 [STAGE 3] CRAG Grader evaluating the reranked documents...")
        valid_docs = []
        
        for i, doc in enumerate(results):
            score = grader_chain.invoke({"query": query, "document": doc.page_content})
            if score.is_relevant == 'yes':
                print(f"   ✅ Doc {i+1}: Relevant")
                valid_docs.append(doc.page_content)
            else:
                print(f"   ❌ Doc {i+1}: Garbage (Discarding)")

        # The Autonomous Pivot
        if not valid_docs:
            print("⚠️ [CRAG] All local documents rejected! Instructing Supervisor to use Web Search.")
            return (
                "INTERNAL SYSTEM DIRECTIVE: The local database does not contain the answer to this query. "
                "The documents retrieved were irrelevant. You MUST immediately use the 'search_web' tool."
            )

        return "\n\n".join(valid_docs)
        
    except Exception as e:
        print(f"\n🚨 [TOOL CRASHED]: {str(e)}") 
        return f"Error executing search: {str(e)}"
    
    
if __name__ == "__main__":
    # Print the name the LLM sees
    print(f"Tool Name: {search_documentation.name}")
    
    # Print the description the LLM reads to make its decision
    print(f"\nTool Description: \n{search_documentation.description}")
    
    # Print the strict JSON schema the LLM must follow to use the tool
    print(f"\nTool Arguments Schema: \n{search_documentation.args}")
