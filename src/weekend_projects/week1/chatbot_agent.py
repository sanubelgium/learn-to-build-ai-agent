#Step1: Setup API Keys for OpenAI
import os
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

# Map the CHATBOT_API_KEY from .env to the standard OPENAI_API_KEY environment variable
if "CHATBOT_API_KEY" in os.environ:
    os.environ["GROQ_API_KEY"] = os.environ["CHATBOT_API_KEY"]


# Initialize the LLM
groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=100
)

#Memory setup
# Keyed by session_id, so concurrent sessions no longer collide.
# NOTE: still in-process memory — fine for a single `uvicorn` worker demo,
# but it resets on restart and won't be shared across multiple workers/
# processes. For that, swap these for a persistent store (SQLite/Redis),
# same idea as chatbot_in_langgraph_with_database.py's SqliteSaver.
store={}
long_term_store={}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id]=ChatMessageHistory()
    return store[session_id]

def save_long_term_memory(session_id: str, input: str, output: str):
    if session_id not in long_term_store:
        long_term_store[session_id] = []
    if len(input) > 20:  # Store inputs longer than 20 characters
        long_term_store[session_id].append(f"User said: {input}")
    if len(long_term_store[session_id]) > 5:  # Keep last 5 chats
        long_term_store[session_id] = long_term_store[session_id][-5:]

def get_long_term_memory(session_id: str):
    return ". ".join(long_term_store.get(session_id, []))

system_prompt = """
You are an intelligent AI assistant with access to a web search tool.

Your primary goal is to provide accurate, factual, and helpful responses.

=========================
GENERAL BEHAVIOR
=========================

- Always answer in a professional, friendly, and concise manner.
- Prefer correctness over confidence.
- Never invent facts.
- If you don't know something and cannot verify it, clearly say so.
- Explain your reasoning only when it helps the user.

=========================
SEARCH STRATEGY
=========================

When using search:

1. Understand the user's intent.

2. Create focused search queries.

3. Search using multiple reputable sources if needed.

4. Prefer official sources over blogs.

Priority:

1. Official websites
2. Government websites
3. Standards organizations
4. Academic institutions
5. Major news organizations
6. Well-known technical documentation
7. Reputable community discussions

=========================
VERIFY INFORMATION
=========================

If multiple sources disagree:

- mention the disagreement
- explain why
- prefer official sources
- avoid presenting uncertain information as fact

=========================
USING SEARCH RESULTS
=========================

Never copy large portions of webpages.

Instead:

- read
- understand
- summarize
- synthesize

Provide a coherent answer rather than a list of copied snippets.

=========================
CITATIONS
=========================

Whenever information comes from search:

- cite the source
- mention the website or organization
- include links if supported

Example:

According to the World Health Organization...

According to Microsoft's official documentation...

=========================
UNCERTAINTY
=========================

If information cannot be verified:

Say:

"I couldn't verify this information from reliable sources."

Never guess.

=========================
MULTIPLE SOURCES
=========================

For important topics:

- compare multiple sources
- identify consensus
- explain differences if they exist

=========================
PRODUCT RECOMMENDATIONS
=========================

When recommending products:

- explain pros and cons
- avoid marketing language
- compare alternatives
- disclose if information is limited

=========================
PROGRAMMING QUESTIONS
=========================

For software development:

Search only if the user asks about:

- latest APIs
- new framework versions
- release notes
- breaking changes
- official documentation

Otherwise answer from existing knowledge.

=========================
NEWS
=========================

For news:

- search first
- identify publication date
- identify publisher
- summarize neutrally
- distinguish facts from opinions

=========================
SAFETY
=========================

Do not fabricate:

- URLs
- research papers
- quotations
- statistics
- legal advice
- medical advice

Encourage consulting professionals for high-risk decisions.

=========================
RESPONSE STYLE
=========================

Structure responses as:

1. Direct answer

2. Supporting explanation

3. Sources (when applicable)

4. Any important caveats

=========================
FINAL RULE
=========================

If web search improves accuracy, use it.

If web search is unnecessary, answer directly without searching.

Always optimize for factual correctness, transparency, and user usefulness.
"""

#Tools Setup
def get_response_from_groq_agent(messages, allow_search, session_id: str):
    """
    Args:
        messages: full chat history for this session, as sent by the client
                  (list of dicts/BaseMessage-like objects with role + content)
        allow_search: whether the agent is allowed to use the web search tool
        session_id: caller-supplied id identifying which conversation this is.
                    Required — each session gets its own isolated history in
                    `store`/`long_term_store`, so concurrent users (or
                    concurrent tabs/threads for the same user) never see or
                    overwrite each other's chat state.
    """
    if not session_id:
        raise ValueError("session_id is required")

    formatted_messages = []
    for msg in messages:
        role = msg.role if hasattr(msg, "role") else msg["role"]
        content = msg.content if hasattr(msg, "content") else msg["content"]       
        formatted_messages.append((role, content))
        
    if not formatted_messages:
        return ""
        
    question = formatted_messages[-1][1]
    print(f"Inside Agent method------------> [session={session_id}] Latest query:", question)

    # Setup session history for THIS session only. We rebuild it from the
    # messages the client sent (rather than trusting whatever's already in
    # `store`), so the client's view of history stays the source of truth —
    # but we no longer touch any other session's history.
    history = get_session_history(session_id)
    history.clear()
    
    # Feed previous messages into the history
    for role, content in formatted_messages[:-1]:
        if role == "user":
            history.add_user_message(content)
        elif role == "assistant":
            history.add_ai_message(content)

    tools = [TavilySearch(max_results=2)] if allow_search else []

    # Agent chain with tools
    agent = create_react_agent(
        model=groq_llm.bind(max_tokens=100),
        tools=tools,
        # prompt=system_prompt
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("system", "Long-term memory: {long_term_memory}"),
        MessagesPlaceholder(variable_name="messages")
    ])
    chain = prompt | agent

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="messages",
    )
        
    response = chain_with_history.invoke(
        {"messages": [HumanMessage(content=question)],"long_term_memory": get_long_term_memory(session_id)},
        config={"configurable": {"session_id": session_id}}
    )
    save_long_term_memory(session_id, question, response['messages'][-1].content)
        
    messages_out = response.get("messages")
    ai_messages = [message.content for message in messages_out if isinstance(message, AIMessage)]
    print(f"\nLong-term Memory [session={session_id}]:")
    print(get_long_term_memory(session_id))
    reply = ai_messages[-1]
    return reply


#Test the AI Agent with a Query
#if __name__ == "__main__":
#    system_prompt = "Act as an AI chatbot who is smart and friendly"
#    query = "Who won the latest Cricket World Cup? Provide the name of the country and the year in which they won."
#    allow_search = True
#    print("\nRunning query with Groq Agent:")
#    ai_response = get_response_from_groq_agent(query, allow_search, system_prompt)
#    print(f"\nResponse from AI: {ai_response}")