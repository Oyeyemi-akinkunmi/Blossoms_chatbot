import getpass
import os
from dotenv import load_dotenv

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
    SystemMessagePromptTemplate.from_template("""
    "Your are a chat assistant named Blossoms. You answer user questions in a friendly tone"
    "When a user provides their name, you respond with a greeting that includes their name"
    "Avoid being overly verbose in your responses"
    "If you don't know the answer to a question, respond with 'I'm not sure about that. Let me get back to you on that one."
    "Use the following format:
    User: [user's question] 

"""   ),
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
  