import uvicorn
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI
from chatbot_agent import get_response_from_groq_agent
from typing import List

load_dotenv()

#Define Req / Res Models for API
class Message(BaseModel):
    role: str
    content: str

class RequestState(BaseModel):
    messages: List[Message]
    allow_search: bool

class ResponseState(BaseModel):
    response: str


#FastAPI Setup
app=FastAPI(title="LangGraph AI Agent")
#API Endpoint to interact with the Chatbot 
@app.post("/chat", response_model=ResponseState)
def chat_endpoint(request: RequestState): 
    print("API called :")
    messages = request.messages
    allow_search = request.allow_search

    # Call AI Agent and Get the final Response 
    response=get_response_from_groq_agent(messages, allow_search)
    print("API response :", response)
    return ResponseState(response=response)

#Run app & Explore Swagger UI Docs
if __name__ == "__main__":
    uvicorn.run("chatbot_api:app", host="127.0.0.1", port=9999, reload=True)