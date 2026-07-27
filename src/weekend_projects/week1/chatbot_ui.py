import streamlit as st
import requests
from dotenv import load_dotenv
load_dotenv()

#Setup UI with streamlit 
st.set_page_config(page_title="Chatbot Agent", layout="centered")
st.title("AI Chatbot Agents")
st.write("Create and Interact with the AI Agents!")
#Input fields from user
system_prompt=st.text_area("Define your AI Agent: ", height=70, placeholder="Type your system prompt here...")
allow_web_search=st.checkbox("Allow Web Search")
user_query=st.text_area("Enter your query: ", height=150, placeholder="Ask Anything!")

#API url for backend
API_URL="http://127.0.0.1:9999/chat"

#Call API on Button Click and Get the Final Response 
if st.button("Lets Chat!"):
    if user_query.strip():
        payload={
            "system_prompt": system_prompt,
            "message": user_query,
            "allow_search": allow_web_search
        }

        response=requests.post(API_URL, json=payload)
        print("response :", response)
        if response.status_code == 200:
            response_data = response.json()
            print("response data :", response_data)
            if "error" in response_data:
                st.error(response_data["error"])
            else:
                #Show Final Response
                st.text_area("Agent Response", value=response_data['response'], height=150, disabled=True)


