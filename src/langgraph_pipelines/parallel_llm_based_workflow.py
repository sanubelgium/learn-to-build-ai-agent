from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
import os
import operator
from dotenv import load_dotenv

load_dotenv()
# Map the CHATBOT_API_KEY from .env to the standard OPENAI_API_KEY environment variable
if "CHATBOT_API_KEY" in os.environ:
    os.environ["GROQ_API_KEY"] = os.environ["CHATBOT_API_KEY"]


#Define evaluation schema
class EvaluationSchema(BaseModel):
    feedback:str = Field(description="Detailed feedback for the essay")
    score:int = Field(description="Score between 0 to 10 for the essay", ge=0,le=10)


# Initialize the LLM
groq_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)
structured_groq_llm = groq_llm.with_structured_output(EvaluationSchema)

#Define the workflow state
class ParallelLLMWorkflowState(TypedDict):
    essay_text:str
    grammar_feedback:str
    clarity_of_thought_feedback:str
    depth_of_analysis_feedback:str
    summerized_feedback:str 
    individual_scores:Annotated[list[int],operator.add]
    overall_essay_score:float

#Define the nodes
def evaluate_grammar(state:ParallelLLMWorkflowState) -> ParallelLLMWorkflowState:
    essay_text = state["essay_text"]
    prompt = f"""Evaluate the language and grammar quality of the essay and provide detailed feedback and assign a score between 0 to 10:
    {essay_text}
    """
    grammar_output = structured_groq_llm.invoke(prompt)
    return {"grammar_feedback": grammar_output.feedback, "individual_scores": [grammar_output.score]}

def evaluate_clarity_of_thought(state:ParallelLLMWorkflowState) -> ParallelLLMWorkflowState:
    essay_text = state["essay_text"]
    prompt = f"""Evaluate the clarity of thought of the essay and provide detailed feedback and assign a score between 0 to 10:
    {essay_text}
    """
    clarity_output = structured_groq_llm.invoke(prompt)
    return {"clarity_of_thought_feedback": clarity_output.feedback, "individual_scores": [clarity_output.score]}

def evaluate_depth_of_analysis(state:ParallelLLMWorkflowState) -> ParallelLLMWorkflowState:
    essay_text = state["essay_text"]
    prompt = f"""Evaluate the depth of analysis of the essay and provide detailed feedback and assign a score between 0 to 10:
    {essay_text}
    """
    depth_output = structured_groq_llm.invoke(prompt)
    return {"depth_of_analysis_feedback": depth_output.feedback, "individual_scores": [depth_output.score]}   


def final_evaluation(state:ParallelLLMWorkflowState) -> ParallelLLMWorkflowState:
    grammar_feedback = state["grammar_feedback"]
    clarity_of_thought_feedback = state["clarity_of_thought_feedback"]
    depth_of_analysis_feedback = state["depth_of_analysis_feedback"]
    prompt = f"""Based on the following feedback from three different evaluators for an essay. 
    You need to create a summerized feedback.
    Here is the feedback from the three evaluators:
    Grammar Feedback: {grammar_feedback}
    Clarity of Thought Feedback: {clarity_of_thought_feedback}
    Depth of Analysis Feedback: {depth_of_analysis_feedback}
    """
    summerized_output = groq_llm.invoke(prompt)

    individual_scores = state["individual_scores"]
    overall_essay_score = sum(individual_scores)/len(individual_scores)

    return {"summerized_feedback": summerized_output.content, "overall_essay_score": overall_essay_score}
    
parallel_graph = StateGraph(ParallelLLMWorkflowState)
#nodes
parallel_graph.add_node("evaluate_grammar",evaluate_grammar) 
parallel_graph.add_node("evaluate_clarity_of_thought",evaluate_clarity_of_thought)
parallel_graph.add_node("evaluate_depth_of_analysis",evaluate_depth_of_analysis)
parallel_graph.add_node("final_evaluation",final_evaluation)
#edges
parallel_graph.add_edge(START,"evaluate_grammar")
parallel_graph.add_edge(START,"evaluate_clarity_of_thought")
parallel_graph.add_edge(START,"evaluate_depth_of_analysis")

parallel_graph.add_edge("evaluate_grammar","final_evaluation")
parallel_graph.add_edge("evaluate_clarity_of_thought","final_evaluation")
parallel_graph.add_edge("evaluate_depth_of_analysis","final_evaluation")

parallel_graph.add_edge("final_evaluation",END)
#compile
parallel_workflow = parallel_graph.compile()
#execute
initial_state = {"essay_text": "**Introduction**\n\nThe RAG system, which stands for Red, Amber, Green, is a color-coded system used to indicate the status or progress of a project, task, or initiative. This system has become an essential tool in various fields, including project management, business, healthcare, and education, as it provides a quick and easy-to-understand visual representation of progress. The importance of the RAG system lies in its ability to simplify complex information, facilitate communication, and enable informed decision-making. By using a simple color-coded system, individuals and teams can quickly identify areas that require attention, allocate resources effectively, and track progress over time.\n\n**History and Origins**\n\nThe RAG system is believed to have originated in the UK in the 1980s, where it was used to report on the status of IT projects. At that time, the system was primarily used to indicate the level of risk associated with a project, with red indicating high risk, amber indicating moderate risk, and green indicating low risk. Over time, the RAG system has evolved to be used in various fields, including business, healthcare, and education. Today, the system is widely used to track progress, identify risks, and allocate resources in a variety of contexts. Despite its widespread adoption, the RAG system remains a relatively simple and intuitive tool, making it accessible to individuals and teams with varying levels of experience and expertise.\n\n**Components of the RAG System**\n\nThe RAG system consists of three colors, each with a distinct meaning and application. These colors are:\n\n* **Red (R)**: Indicates a critical or high-risk status, often requiring immediate attention. Examples of situations that might be classified as red include project delays, budget overruns, or significant issues that require immediate resolution. In a project management context, a red status might indicate that a project is at risk of missing its deadline or exceeding its budget.\n* **Amber (A)**: Indicates a cautionary or warning status, requiring monitoring and potential intervention. Examples of situations that might be classified as amber include project risks, minor issues, or potential problems that require attention but are not yet critical. In a business context, an amber status might indicate that a particular product or service is experiencing declining sales or customer satisfaction.\n* **Green (G)**: Indicates a healthy or on-track status, with no significant issues or risks. Examples of situations that might be classified as green include projects progressing as planned, with no major issues or concerns. In a healthcare context, a green status might indicate that a patient is recovering well from a procedure or treatment.\n\n**Applications of the RAG System**\n\nThe RAG system has a wide range of applications across various fields, including:\n\n* **Project Management**: The RAG system is used to track project progress, identify risks, and allocate resources. By using the RAG system, project managers can quickly identify areas that require attention and take proactive steps to mitigate risks and ensure project success.\n* **Business**: The RAG system is used to monitor business performance, identify areas for improvement, and make informed decisions. By using the RAG system, business leaders can quickly identify areas that require attention and take proactive steps to address issues and capitalize on opportunities.\n* **Healthcare**: The RAG system is used to track patient progress, identify potential health risks, and allocate resources. By using the RAG system, healthcare professionals can quickly identify patients who require additional attention or care and take proactive steps to prevent complications and improve outcomes.\n* **Education**: The RAG system is used to track student progress, identify areas for improvement, and allocate resources. By using the RAG system, educators can quickly identify students who require additional support or attention and take proactive steps to provide targeted interventions and improve student outcomes.\n\n**Benefits of the RAG System**\n\nThe RAG system offers a number of benefits, including:\n\n* **Simplified Communication**: The RAG system provides a clear and concise visual representation of status or progress, making it easy to communicate complex information to stakeholders.\n* **Improved Decision-Making**: The RAG system enables informed decision-making by providing a quick and easy-to-understand overview of status or progress.\n* **Enhanced Collaboration**: The RAG system facilitates collaboration and communication among team members and stakeholders, ensuring that everyone is on the same page and working towards common goals.\n* **Increased Efficiency**: The RAG system saves time and resources by providing a quick and easy-to-understand visual representation of status or progress, allowing individuals and teams to focus on high-priority tasks and activities.\n\n**Limitations and Challenges**\n\nWhile the RAG system is a valuable tool, it is not without its limitations and challenges. Some of the key limitations and challenges include:\n\n* **Subjectivity**: The RAG system can be subjective, with different individuals or teams interpreting the colors differently. This can lead to confusion and inconsistencies, particularly if the criteria for each color are not clearly defined.\n* **Lack of Standardization**: There is no standardized definition or application of the RAG system, which can lead to confusion and inconsistencies. This can make it difficult to compare progress or status across different projects or initiatives.\n* **Over-Reliance on Colors**: The RAG system can be over-relied upon, leading to a lack of attention to underlying issues or details. This can result in a superficial understanding of progress or status, rather than a deeper understanding of the underlying factors that are driving results.\n\n**Best Practices for Implementing the RAG System**\n\nTo get the most out of the RAG system, it is essential to follow best practices for implementation. Some of the key best practices include:\n\n* **Define Clear Criteria**: Establish clear criteria for each color to ensure consistency and accuracy. This should include specific metrics or indicators that will be used to determine the color for each project or initiative.\n* **Establish a Common Understanding**: Ensure that all team members and stakeholders have a common understanding of the RAG system and its application. This should include training and education on the use of the RAG system, as well as regular review and feedback.\n* **Regularly Review and Update**: Regularly review and update the RAG system to ensure it remains relevant and effective. This should include ongoing evaluation of the criteria and metrics used to determine the color for each project or initiative, as well as adjustments to the system as needed.\n\n**Conclusion**\n\nIn conclusion, the RAG system is a widely used and effective tool for indicating status or progress. By providing a simple and intuitive color-coded system, the RAG system enables individuals and teams to quickly identify areas that require attention, allocate resources effectively, and track progress over time. While the RAG system has its limitations and challenges, it can be a valuable tool for project management, business, healthcare, and education, among other fields. By following best practices for implementation and ongoing review and evaluation, individuals and teams can get the most out of the RAG system and achieve their goals. Ultimately, the RAG system is a powerful tool that can help individuals and teams to simplify complex information, facilitate communication, and drive success."}
final_state = parallel_workflow.invoke(initial_state)
print("summerized_feedback: ", final_state["summerized_feedback"])
print("overall_essay_score: ", final_state["overall_essay_score"])
print(final_state)