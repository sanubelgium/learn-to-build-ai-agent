from langgraph.graph import StateGraph, START, END
from typing import TypedDict

#define state
class BMIState(TypedDict):
    weight_kg: float
    height_cms: float
    bmi: float
    category: str

def calculate_bmi(state: BMIState) -> BMIState:
    """Calculates BMI from weight and height"""
    weight_kg = state['weight_kg']
    height_cms = state['height_cms']
    bmi = weight_kg / ((height_cms/100)**2)
    state['bmi'] = round(bmi, 2)
    return state

def categorize_bmi(state: BMIState) -> BMIState:
    """Categorizes BMI from weight and height"""
    bmi = state['bmi']
    if bmi < 18.5:
        state['category'] = "Underweight"
    elif bmi < 25:
        state['category'] = "Normal weight"
    elif bmi < 30:
        state['category'] = "Overweight"
    else:
        state['category'] = "Obesity"
    return state

#Define Graph
builderGraph = StateGraph(BMIState)

#Add nodes to graph
builderGraph.add_node("calculate_bmi", calculate_bmi)
builderGraph.add_node("categorize_bmi", categorize_bmi)

#Add Edges to graph
builderGraph.add_edge(START, "calculate_bmi")
builderGraph.add_edge("calculate_bmi", "categorize_bmi")
builderGraph.add_edge("categorize_bmi", END)

#Compile graph
bmi_workflow = builderGraph.compile()

#Execute graph
initial_state = {"weight_kg": 76, "height_cms": 178}
final_state = bmi_workflow.invoke(initial_state)
print("final_state: ", final_state)

#Test graph using stream
print("\nTesting graph using stream...")
for event in bmi_workflow.stream(initial_state):
    print(event)

