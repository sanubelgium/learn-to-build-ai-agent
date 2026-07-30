import streamlit as st
import requests
from dotenv import load_dotenv
load_dotenv()

# Setup UI with streamlit 
st.set_page_config(
    page_title="AI Chatbot Agent",
    layout="centered"
)


allow_web_search = st.sidebar.checkbox("Allow Web Search", value=False)


# Add a clear chat button in sidebar
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

st.title("AI Chatbot Agent")
st.write("Interact with your custom-defined AI Agent in real time! Enjoy the chat.")

# API url for backend
API_URL = "http://127.0.0.1:9999/chat"

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if user_query := st.chat_input("Ask me anything right now"):
    # Display user message in chat message container
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            payload = {
                "messages": st.session_state.messages,
                "allow_search": allow_web_search
            }
            try:
                response = requests.post(API_URL, json=payload)
                if response.status_code == 200:
                    response_data = response.json()
                    
                    if "error" in response_data:
                        st.error(response_data["error"])
                    else:
                        reply = response_data['response']
                        message_placeholder.markdown(reply)
                        st.session_state.messages.append({"role": "assistant", "content": reply})
                else:
                    st.error(f"Error {response.status_code}: Could not get response.")
            except Exception as e:
                st.error(f"Could not connect to backend server: {e}")



