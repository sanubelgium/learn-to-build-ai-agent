import os
from langgraph.graph import StateGraph, START, END
from  typing import TypedDict
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
# Map the CHATBOT_API_KEY from .env to the standard OPENAI_API_KEY environment variable
if "CHATBOT_API_KEY" in os.environ:
    os.environ["GROQ_API_KEY"] = os.environ["CHATBOT_API_KEY"]

# Initialize the LLM
    groq_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

#Define the workflow state
class PromptChainingWorkflowState(TypedDict):
    topic:str
    outline:str
    content:str
    evaluation:str
    

def create_outline(state:PromptChainingWorkflowState) -> PromptChainingWorkflowState:
    prompt = f"""Generate a detailed outline for the topic: {state['topic']}"""
    outline = groq_llm.invoke(prompt).content
    state["outline"] = outline
    return state

def generate_content(state:PromptChainingWorkflowState) -> PromptChainingWorkflowState:
    prompt = f"""Generate detailed content based on the topic: {state['topic']} follow this outline: \n {state['outline']}"""
    content = groq_llm.invoke(prompt).content
    state["content"] = content
    return state

def evaluate_blog(state:PromptChainingWorkflowState) -> PromptChainingWorkflowState:
    prompt = f"""Evaluate the following blog content for quality and accuracy:\n {state['content']} based on this outline {state['outline']}. Provide suggestions for improvement.\n You can either accept or reject the blog."""
    evaluation = groq_llm.invoke(prompt).content
    state["evaluation"] = evaluation
    return state 
 
 #Create Graph
prompt_chaining_graph = StateGraph(PromptChainingWorkflowState)
#Adding Nodes
prompt_chaining_graph.add_node("create_outline",create_outline)
prompt_chaining_graph.add_node("generate_content",generate_content)
prompt_chaining_graph.add_node("evaluate_blog",evaluate_blog)
#Adding Edges
prompt_chaining_graph.add_edge(START,"create_outline")
prompt_chaining_graph.add_edge("create_outline","generate_content")
prompt_chaining_graph.add_edge("generate_content","evaluate_blog")
prompt_chaining_graph.add_edge("evaluate_blog",END)
#Compiling graph
prompt_chaining_workflow = prompt_chaining_graph.compile()
#Execute graph
initial_state = {"topic": "Rise of AI with NVIDIA"}
final_state = prompt_chaining_workflow.invoke(initial_state)
print("outline: ", final_state["outline"])   
print("content: ", final_state["content"])   
print("evaluation: ", final_state["evaluation"]) 