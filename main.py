import getpass
import os
from dotenv import load_dotenv
from routers import rag

load_dotenv()
if not os.environ.get('groq_api_key'):
    os.environ['groq_api_key'] = getpass.getpass('Enter your Groq API key: ')

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver

from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI()

class chat(BaseModel):
    text:str

model = ChatGroq(model="openai/gpt-oss-20b", api_key=os.environ['groq_api_key'], temperature=0)

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
       """ You are a chat assistant named Blossoms. 
            Answer user questions in a friendly, natural, and chatty tone. 
            Avoid being overly verbose in your responses. 
            Use the user's name only occasionally when it feels natural, not in every response. 
            If you don't know the answer to a question, respond with: 
            "I'm not sure about that. Let me get back to you on that one."
        """
       ),
    HumanMessagePromptTemplate.from_template("{text}"
    )
])

def call_model(state:dict):
    response = model.invoke(state["messages"])
    return {"messages":response}

workflow = StateGraph(state_schema=MessagesState)

memory = MemorySaver()
workflow.add_node("model", model)
workflow.add_node("call_model", call_model)

workflow.add_edge(START, "call_model")
workflow.add_edge("call_model", END)

graph = workflow.compile(checkpointer=memory)

@app.get("/")
async def root(text:str):
    messages = prompt.format_messages(
        text=text)
    response = graph.invoke(
        {"messages":messages},
        config={
            "configurable": {"thread_id": "user-1"}
        })
    ai_res = response["messages"][-1]
    return {"response":ai_res.content}
  
app.include_router(rag.router)