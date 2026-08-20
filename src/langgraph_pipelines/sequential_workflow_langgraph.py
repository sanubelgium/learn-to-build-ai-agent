from langgraph.graph import StateGraph, START, END
from  typing import TypedDict
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
# Map the CHATBOT_API_KEY from .env to the standard OPENAI_API_KEY environment variable
if "CHATBOT_API_KEY" in os.environ:
    os.environ["GROQ_API_KEY"] = os.environ["CHATBOT_API_KEY"]

# Initialize the LLM
groq_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

#Define the workflow state
class SimpleSequentialGraphState(TypedDict):
    question:str 
    answer:str   

def ask_llm(state:SimpleSequentialGraphState) -> SimpleSequentialGraphState:
    question = state["question"]
    prompt = f"""You are a helpful assistant. Answer the following question:
    {question}
    """
    result = groq_llm.invoke(prompt).content
    state["answer"] = result
    return state

#Create Graph
simple_graph = StateGraph(SimpleSequentialGraphState)
#Adding Nodes
simple_graph.add_node("ask_llm",ask_llm)
#Adding Edges
simple_graph.add_edge(START,"ask_llm")
simple_graph.add_edge("ask_llm",END)
#Compiling graph
simple_workflow = simple_graph.compile()
#Execute graph
initial_state = {"question": "What is RAG?"}
final_state = simple_workflow.invoke(initial_state)
print("final_state: ", final_state)


