import streamlit as st
from chatbot_using_langgraph import chatbot_workflow
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import uuid
# load .env variables
load_dotenv()

# Initialize session state for chat_history
if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []
    #autogenerate thread id
    st.session_state['thread_id'] = str(uuid.uuid4())

# Display chat chat_history from history on app rerun
for chat_session in st.session_state['chat_history']:
    with st.chat_message(chat_session["role"]):
        st.markdown(chat_session["content"])

CONFIG_DICT={"configurable":{"thread_id":st.session_state['thread_id']}}
#chatting starts here
user_query = st.chat_input("Ask me anything right now")
if user_query:
    st.session_state['chat_history'].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    initial_state = {"conversations": [HumanMessage(content=user_query)]}
    with st.chat_message("assistant"):
        ai_messages = st.write_stream(
            message_chunk.content for message_chunk,metadata in chatbot_workflow.stream(
                initial_state,
                config=CONFIG_DICT,
                stream_mode="messages"
            ) 
        )
    st.session_state['chat_history'].append({"role": "assistant", "content": ai_messages})