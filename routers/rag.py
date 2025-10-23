from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, TypedDict
import os, tempfile
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# LangChain & Pinecone imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langgraph.graph import StateGraph, START
from langchain_pinecone import PineconeVectorStore

# ---------------------------------------------------

load_dotenv()

# Initialize LLM
llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.environ["groq_api_key"], temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Initialize Pinecone
pc= Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = os.environ["PINECONE_INDEX_NAME"]

# Create index if it doesn’t exist
if index_name not in [i.name for i in pc.list_indexes()]:
    pc.create_index(name=os.environ["PINECONE_INDEX_NAME"],
                     dimension=768, 
                     metric="cosine",
                     spec=ServerlessSpec(cloud='aws', region=os.environ.get("PINECONE_ENVIRONMENT", "us-east-1")))

# Initialize VectorStore
vs = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# Define LangGraph State
class State(TypedDict):
    q: str
    answer: str
    docs: List[Document]

# Prompt
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a helpful assistant that helps people find information. "
        "If you cannot find the answer, just say 'I don't know'."
    ),
    HumanMessagePromptTemplate.from_template("{q}\n\n{docs}"),
])

# Retrieval
def retrieve_docs(state: State):
    retrieved = vs.similarity_search(state["q"], k=4)
    return {"docs": retrieved}

# Generation
def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["docs"])
    messages = prompt.invoke({"q": state["q"], "docs": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Build graph
graph_builder = StateGraph(State).add_sequence([retrieve_docs, generate])
graph_builder.add_edge(START, "retrieve_docs")
graph = graph_builder.compile()

# FastAPI Router
router = APIRouter(prefix="/pdf_chat", tags=["PDF Chatbot"])

# Request model
class QueryRequest(BaseModel):
    question: str

# Ask endpoint
@router.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        response = graph.invoke({"q": request.question})
        return {"question": request.question, "answer": response["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
'''
#Upload endpoint
@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(await file.read())

        # Load and split the PDF
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        splitted_docs = splitter.split_documents(docs)

        # Add to Pinecone
        PineconeVectorStore.from_documents(
            documents=splitted_docs,
            embedding=embeddings,
            index_name=index_name
        )

        os.remove(tmp_path)
        return {"message": f"Successfully indexed {file.filename} into Pinecone."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")
'''