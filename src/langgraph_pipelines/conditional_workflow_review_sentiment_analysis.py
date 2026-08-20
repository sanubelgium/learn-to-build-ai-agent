from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
import os
import operator
from dotenv import load_dotenv

load_dotenv()
# Map the CHATBOT_API_KEY from .env to the standard OPENAI_API_KEY environment variable
if "CHATBOT_API_KEY" in os.environ:
    os.environ["GROQ_API_KEY"] = os.environ["CHATBOT_API_KEY"]

# Initialize the LLM
groq_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

#Define evaluation schema
class SentimentSchema(BaseModel):
    sentiment:Literal["positive", "negative"] = Field(description="Sentiment of the review: positive, negative")

class DiagnosisSchema(BaseModel):
    issue_type:Literal["UX", "Performance", "Bug", "Support", "Feature", "Other"] = Field(description="Category of issue mentioned in the review")
    tone:Literal["Dissapointed", "Angry", "Frustrated", "Calm", "Informative"] = Field(description="Emotional tone expressed by the user in the review")
    urgency:Literal["Critical", "High", "Medium", "Low"] = Field(description="Urgency of the issue mentioned in the review")

structured_groq_llm = groq_llm.with_structured_output(SentimentSchema)
structured_groq_llm_for_diagnosis = groq_llm.with_structured_output(DiagnosisSchema)

#define workflow state
class ReviewSentimentWorkflowState(TypedDict):
    review_text:str
    sentiment:Literal["positive", "negative"]
    diagonosis:dict
    response:str

def analyse_sentiment(state:ReviewSentimentWorkflowState) -> ReviewSentimentWorkflowState:
    review_text = state["review_text"]
    prompt = f"""Analyse the sentiment of the review and provide detailed feedback as positive or negative:
    {review_text}
    """
    sentiment_output = structured_groq_llm.invoke(prompt)
    return {"sentiment": sentiment_output.sentiment} 

def check_sentiment(state:ReviewSentimentWorkflowState) -> Literal["positive_repsonse", "run_diagonosis"]:
    sentiment = state["sentiment"]
    if sentiment == "positive":
        return "positive_repsonse"
    else:
        return "run_diagonosis"

def positive_response(state:ReviewSentimentWorkflowState) -> ReviewSentimentWorkflowState:
    review_text = state["review_text"]
    prompt = f"""Write a warm and appreciative feedback and thank the customer for the review:
    {review_text}
    Also kindly ask user to share the review on other social media platform if they are willing to do so.
    """
    response = groq_llm.invoke(prompt)
    return {"response": response.content}

def run_diagonosis(state:ReviewSentimentWorkflowState) -> ReviewSentimentWorkflowState:
    review_text = state["review_text"]
    prompt = f"""Run a detailed diagnosis on this negative feedback:
    {review_text}
    Return issue type, tone and urgency.
    """
    diagnosis = structured_groq_llm_for_diagnosis.invoke(prompt)
    return {"diagonosis": diagnosis.model_dump()}

def negative_response(state:ReviewSentimentWorkflowState) -> ReviewSentimentWorkflowState:
    review_text = state["review_text"]
    diagnosis = state["diagonosis"]
    prompt = f"""You are a support assistant. User had a {diagnosis['issue_type']} sounded {diagnosis['tone']} tone and with a {diagnosis['urgency']} urgency.
    Respond a empathetic, helpfull and professional response to the user.
    """
    response = groq_llm.invoke(prompt)
    return {"response": response.content}

review_sentiment_graph = StateGraph(ReviewSentimentWorkflowState)
#node
review_sentiment_graph.add_node("analyse_sentiment",analyse_sentiment)
review_sentiment_graph.add_node("positive_repsonse",positive_response)  
review_sentiment_graph.add_node("run_diagonosis",run_diagonosis)
review_sentiment_graph.add_node("negative_repsonse",negative_response)

#edges
review_sentiment_graph.add_edge(START,"analyse_sentiment")
review_sentiment_graph.add_conditional_edges("analyse_sentiment",check_sentiment)

review_sentiment_graph.add_edge("positive_repsonse",END)

review_sentiment_graph.add_edge("run_diagonosis","negative_repsonse")
review_sentiment_graph.add_edge("negative_repsonse",END)

#compile
review_sentiment_workflow = review_sentiment_graph.compile()

#execute
initial_state = {"review_text":"I am experiencing issues with the app crashing frequently. It is very frustrating."}
result = review_sentiment_workflow.invoke(initial_state)
print(result)
