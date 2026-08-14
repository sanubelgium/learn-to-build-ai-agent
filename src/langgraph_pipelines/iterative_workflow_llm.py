from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Annotated
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import os
import operator
from dotenv import load_dotenv

load_dotenv()
# Map the CHATBOT_API_KEY from .env to the standard OPENAI_API_KEY environment variable
if "CHATBOT_API_KEY" in os.environ:
    os.environ["GROQ_API_KEY"] = os.environ["CHATBOT_API_KEY"]

# Initialize the LLM
generator_model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.8
)
optimizer_model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.8
)
evaluator_model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

#state
class PostState(TypedDict):
    topic:str
    post:str
    evaluation:Literal["Approved","Not Approved"]
    feedback:str
    iteration:int
    max_iterations:int
    post_history:Annotated[list[str],operator.add]
    feedback_history:Annotated[list[str],operator.add]

#Schema
class EvaluationSchema(BaseModel):
    evaluation:Literal["Approved","Not Approved"] = Field(..., description="Evaluation of the post")
    feedback:str = Field(..., description="Feedback for the generated post")

structured_evaluator_model = evaluator_model.with_structured_output(EvaluationSchema)

#generation node
def generate_post(state:PostState) -> PostState:
    generation_messages = [
    SystemMessage(content="You are a funny and clever Twitter/X influencer."),
    HumanMessage(content=f"""
    Write a short, original, and hilarious tweet on the topic: "{state['topic']}".

    Rules:
    - Do NOT use question-answer format.
    - Max 280 characters.
    - Use observational humor, irony, sarcasm, or cultural references.
    - Think in meme logic, punchlines, or relatable takes.
    - Use simple, day to day english.
    """)
    ]
    generated_post = generator_model.invoke(generation_messages).content
    return {"post": generated_post,"post_history":[generated_post]}

#evaluatio node    
def evaluate_post(state:PostState) -> PostState:
    evaluation_messages = [
    SystemMessage(content="You are a ruthless, no-laugh-given Twitter critic. You evaluate tweets based on humor, originality, virality, and tweet format."),
    HumanMessage(content=f"""
Evaluate the following post:

Post: "{state['post']}"

Use the criteria below to evaluate the post:

1. Originality - Is this fresh, or have you seen it a hundred times before?
2. Humor - Did it genuinely make you smile, laugh, or chuckle?
3. Punchiness - Is it short, sharp, and scroll-stopping?
4. Virality Potential - Would people retweet or share it?
5. Format - Is it a well-formed tweet (not a setup-punchline joke, not a Q&A joke, and under 280 characters)?

Auto-reject if:
- It's written in question-answer format (e.g., "Why did..." or "What happens when...")
- It exceeds 280 characters
- It reads like a traditional setup-punchline joke
- Dont end with generic, throwaway, or deflating lines that weaken the humor (e.g., "Masterpieces of the auntie-uncle universe" or vague summaries)

### Respond ONLY in structured format:
- evaluation: "approved" or "needs_improvement"
- feedback: One paragraph explaining the strengths and weaknesses
""")
]
    evaluation_response = structured_evaluator_model.invoke(evaluation_messages)
    return {"evaluation": evaluation_response.evaluation, "feedback": evaluation_response.feedback,"feedback_history":[evaluation_response.feedback]}

#optimization node
def optimize_post(state:PostState) -> PostState:
    optimization_messages = [
    SystemMessage(content="You punch up tweets for virality and humor based on given feedback."),
    HumanMessage(content=f"""
Improve the tweet based on this feedback:
"{state['feedback']}"

Topic: "{state['topic']}"
Original Tweet:
{state['post']}

Re-write it as a short, viral-worthy tweet. Avoid Q&A style and stay under 280 characters.
""")
]
    optimized_post = optimizer_model.invoke(optimization_messages).content
    iteration = state["iteration"] + 1
    return {"post": optimized_post, "iteration": iteration, "post_history":[optimized_post]}
    


def route_based_on_condition(state:PostState):
    if state["evaluation"]=="Approved" or state["iteration"]>state["max_iterations"]:
        return "approved"
    else:
        return "need_approval"

#graph
post_graph=StateGraph(PostState)

#nodes
post_graph.add_node("generate_post",generate_post)
post_graph.add_node("optimize_post",optimize_post)
post_graph.add_node("evaluate_post",evaluate_post)

#edges
post_graph.add_edge(START, "generate_post")
post_graph.add_edge("generate_post", "evaluate_post")
post_graph.add_conditional_edges("evaluate_post", route_based_on_condition,{"approved":END,"need_approval":"optimize_post"})
post_graph.add_edge("optimize_post", "evaluate_post")
#compile
post_workflow = post_graph.compile()
#execute
initial_state = {
    "topic": "Generate a weak topic",
    "iteration":1,
    "max_iterations": 3
}
output = post_workflow.invoke(initial_state)
print(output)


