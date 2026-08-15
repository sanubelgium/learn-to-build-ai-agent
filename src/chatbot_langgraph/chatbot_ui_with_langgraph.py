import streamlit as st
from chatbot_using_langgraph import chatbot_workflow
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import uuid
# load .env variables
load_dotenv()

#Methods
def generate_thread_id():
    return str(uuid.uuid4())

def create_new_chat():
    st.session_state['chat_history'] = []
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])

def add_thread(thread_id):
    if thread_id not in st.session_state['all_threads']:
        st.session_state['all_threads'].append(thread_id)
def retrieve_chat_history(thread_id):
    CONFIG_PER_THREAD={"configurable":{"thread_id":thread_id}}
    chat_history_per_thread = chatbot_workflow.get_state(config=CONFIG_PER_THREAD).values['conversations']
    return chat_history_per_thread


# Initializations
if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []
if "thread_id" not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()
if "all_threads" not in st.session_state:
    st.session_state['all_threads'] = [st.session_state['thread_id']]
if "conversation" not in st.session_state:
    st.session_state['conversation'] = []
if "selected_thread_id" not in st.session_state:
    st.session_state['selected_thread_id'] = st.session_state['thread_id']    
add_thread(st.session_state['thread_id'])

#============================================ Sidebar UI ====================================================
st.sidebar.title("Chat History")
st.sidebar.header("Conversations")
#new chat button
if st.sidebar.button("New Chat"):
    create_new_chat()
#loop through each thread id and display it
for thread in st.session_state['all_threads']:
    if st.sidebar.button(thread):
        st.session_state['thread_id'] = thread
        reterieved_converstaion = retrieve_chat_history(thread)
        #convert the retrievered conversation to the format of the chat_history
        st.session_state['chat_history'] = []
        for message in reterieved_converstaion:
            st.session_state['chat_history'].append({"role": message.type, "content": message.content})
        
#============================================ Main Chat UI ====================================================
# Display chat chat_history from history on app rerun
for chat_history in st.session_state['chat_history']:
    with st.chat_message(chat_history["role"]):
        st.text(chat_history["content"])
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

