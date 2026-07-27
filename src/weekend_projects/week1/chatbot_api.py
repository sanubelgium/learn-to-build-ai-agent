import uvicorn
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI
from chatbot_agent import get_response_from_groq_agent

load_dotenv()

#Define Req / Res Models for API
class RequestState(BaseModel):
    system_prompt: str
    message: str
    allow_search: bool

class ResponseState(BaseModel):
    response: str


#FastAPI Setup
app=FastAPI(title="LangGraph AI Agent")
#API Endpoint to interact with the Chatbot 
@app.post("/chat", response_model=ResponseState)
def chat_endpoint(request: RequestState): 
    print("API called :")
    system_prompt = request.system_prompt
    query = request.message
    allow_search = request.allow_search

    # Call AI Agent and Get the final Response 
    response=get_response_from_groq_agent(query, allow_search, system_prompt)
    print("API response :", response)
    return ResponseState(response=response)

#Run app & Explore Swagger UI Docs
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9999)