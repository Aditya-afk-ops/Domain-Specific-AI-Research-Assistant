import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- 1. Load Environment Variables ---
load_dotenv()

# --- 2. Configuration ---
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
EMBEDDINGS_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# --- 3. Initialize FastAPI App ---
app = FastAPI(
    title="Domain-Specific AI Research Assistant",
    description="An API for querying research documents using RAG.",
    version="1.0.0"
)

# --- 4. Defining Pydantic Models (for Request and Response) ---
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    context: str

# --- 5. Setting up the RAG Pipeline (LangChain) ---

# Initializing the components
try:
    # Initializing Embeddings
    embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)
    
    # Initializing the LLM
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    # Connecting to the existing Pinecone vector store
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )
    
    # Creating the retriever, which fetches relevant documents
    #"Semantic Retrieval" part
    retriever = vectorstore.as_retriever()

    # Defining the prompt template
    template = """
    You are an expert research assistant. Use the following pieces of retrieved context 
    to answer the user's question.
    If you don't know the answer, just say that you don't know. 
    Keep the answer concise and based *only* on the provided context.

    CONTEXT: 
    {context}

    QUESTION: 
    {question}

    ANSWER:
    """

    prompt = PromptTemplate.from_template(template)

    # Creating the RAG chain using LangChain Expression Language (LCEL)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # A separate chain to get the context (for the response)
    context_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
    )

    print("RAG Pipeline initialized successfully.")

except Exception as e:
    print(f"Error initializing RAG pipeline: {e}")
    rag_chain = None
    context_chain = None

# --- 6. Defining the FastAPI Endpoint ---

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Receives a question, retrieves context, and generates an answer.
    """
    if not rag_chain or not context_chain:
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized.")

    question = request.question

    # 1. Get the context
    context_data = await context_chain.ainvoke(question)
    
    # Format context for the response
    retrieved_docs = context_data.get("context", [])
    context_str = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # 2. Get the answer
    # We use 'ainvoke' for asynchronous execution (FastAPI 'async def')
    answer = await rag_chain.ainvoke(question)
    
    return QueryResponse(answer=answer, context=context_str)

@app.get("/")
def read_root():
    return {"message": "AI Research Assistant API is running."}

# --- 7. Run the API (for local development) ---
if __name__ == "__main__":
    import uvicorn
    # Uvicorn is the server that runs FastAPI
    # We use 8080 because it's the default for Google Cloud Run
    uvicorn.run(app, host="0.0.0.0", port=8080)