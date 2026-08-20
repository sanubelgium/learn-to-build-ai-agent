from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Annotated
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import os
import operator
from dotenv import load_dotenv
import sqlite3

load_dotenv()
# Map the CHATBOT_API_KEY from .env to the standard OPENAI_API_KEY environment variable
if "CHATBOT_API_KEY" in os.environ:
    os.environ["GROQ_API_KEY"] = os.environ["CHATBOT_API_KEY"]

# Initialize the LLM
chat_model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.8
)
#state
class ChatState(TypedDict):
    conversations:Annotated[list[BaseMessage],add_messages]

#node functions
def chat_node(state:ChatState) -> ChatState:
    query = state["conversations"]
    response = chat_model.invoke(query)
    return {"conversations":[response]}
#checkpoint
# checkpoint=MemorySaver()
sqlite_connection=sqlite3.connect("converstaion_history.db",check_same_thread=False)
checkpointer=SqliteSaver(conn=sqlite_connection)
#graph
chatbot_graph = StateGraph(ChatState)

#nodes
chatbot_graph.add_node("chat_node",chat_node)

#edges
chatbot_graph.add_edge(START,"chat_node")
chatbot_graph.add_edge("chat_node",END)
#compile
chatbot_workflow = chatbot_graph.compile(checkpointer=checkpointer)

def find_all_thread_ids():
    all_threads=set()
    for checkpoint in checkpointer.list(None):
        thread_id=checkpoint.config["configurable"]["thread_id"]
        all_threads.add(thread_id)
    return list(all_threads)
        