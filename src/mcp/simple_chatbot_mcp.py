from langgraph.graph import StateGraph, START
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
import asyncio
import os
import sys
from langchain_mcp_adapters.client import MultiServerMCPClient


load_dotenv()  # Load environment variables from .env file
# Map the CHATBOT_API_KEY from .env to the standard OPENAI_API_KEY environment variable
if "CHATBOT_API_KEY" in os.environ:
    os.environ["GROQ_API_KEY"] = os.environ["CHATBOT_API_KEY"]

# Initialize the LLM
groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


#MCP client for local FastMCP server 
mcp_client = MultiServerMCPClient(
    {
        "calculator": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["./src/mcp/tools/calculator_mcp_server.py"]
        }
    }
)

# state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

#methods
async def build_graph():
    #tools
    tools = await mcp_client.get_tools()
    print("tools",tools)
    llm_with_tools = groq_llm.bind_tools(tools)
    # nodes
    async def chat_node(state: ChatState):
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {'messages': [response]}

    tool_node = ToolNode(tools)

    # defining graph and nodes
    mcp_graph = StateGraph(ChatState)

    mcp_graph.add_node("chat_node", chat_node)
    mcp_graph.add_node("tools", tool_node)

    # defining graph connections
    mcp_graph.add_edge(START, "chat_node")
    mcp_graph.add_conditional_edges("chat_node", tools_condition)
    mcp_graph.add_edge("tools", "chat_node")

    mcp_chatbot = mcp_graph.compile()
    return mcp_chatbot


# running the graph
async def main():
    mcp_chatbot = await build_graph()
    result = await mcp_chatbot.ainvoke({"messages": [HumanMessage(content="Find the modulus of 13 and 5, give answer like a cricket commentator.")]})
    print(result['messages'][-1].content)
    
    
if __name__ == "__main__":
    asyncio.run(main())