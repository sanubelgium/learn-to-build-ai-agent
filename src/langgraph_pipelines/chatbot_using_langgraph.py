from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Annotated
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
import os
import operator
from dotenv import load_dotenv

load_dotenv()
# Map the CHATBOT_API_KEY from .env to the standard OPENAI_API_KEY environment variable
if "CHATBOT_API_KEY" in os.environ:
    os.environ["GROQ_API_KEY"] = os.environ["CHATBOT_API_KEY"]

# Initialize the LLM
chat_model = ChatGroq(
    model="llama-3.3-70b-versatile",
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

#graph
chatbot_graph = StateGraph(ChatState)

#nodes
chatbot_graph.add_node("chat_node",chat_node)

#edges
chatbot_graph.add_edge(START,"chat_node")
chatbot_graph.add_edge("chat_node",END)
#compile
chatbot_workflow = chatbot_graph.compile()
#execute
while True:
    user_message=input("User: ")
    if user_message.strip().lower() in ["exit","bye"]:
        print("Exiting Chatbot...")
        break
    initial_state = {"conversations": [HumanMessage(content=user_message)]}
    output = chatbot_workflow.invoke(initial_state)
    print(output["conversations"][-1].content)