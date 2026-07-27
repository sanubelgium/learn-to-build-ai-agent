#Step1: Setup API Keys for OpenAI
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

load_dotenv()

# Map the CHATBOT_API_KEY from .env to the standard OPENAI_API_KEY environment variable
if "CHATBOT_API_KEY" in os.environ:
    os.environ["GROQ_API_KEY"] = os.environ["CHATBOT_API_KEY"]

# Initialize the LLM
groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=4096
)
#Tools Setup
def get_response_from_groq_agent(query, allow_search, system_prompt):
    tools = [TavilySearch(max_results=2)] if allow_search else []
    agent = create_agent(
        model=groq_llm,
        tools=tools,
        system_prompt=system_prompt
    )
    print("Inside Agent method------------> ", query)
    state = {"messages": [("user", query)]}
    response = agent.invoke(state)
    messages = response.get("messages")
    ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]
    print("Inside Agent method end ------------> ", ai_messages[-1])
    return ai_messages[-1]


#Test the AI Agent with a Query
#if __name__ == "__main__":
#    system_prompt = "Act as an AI chatbot who is smart and friendly"
#    query = "Who won the latest Cricket World Cup? Provide the name of the country and the year in which they won."
#    allow_search = True
#    print("\nRunning query with Groq Agent:")
#    ai_response = get_response_from_groq_agent(query, allow_search, system_prompt)
#    print(f"\nResponse from AI: {ai_response}")