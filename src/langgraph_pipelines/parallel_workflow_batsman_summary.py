from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class BatsmanState(TypedDict):
    runs: int
    balls_faced: int
    fours: int
    sixes: int
    strike_rate: float
    balls_per_boundary: float
    boundary_percentage: float
    summary: str

def strike_rate(state: BatsmanState) -> BatsmanState:
    """Calculates strike rate from runs and balls faced"""
    runs = state['runs']
    balls_faced = state['balls_faced']
    strike_rate = (runs / balls_faced) * 100
    return {"strike_rate": strike_rate}

def balls_per_boundary(state: BatsmanState) -> BatsmanState:
    """Calculates balls per boundary from runs and balls faced"""

    balls_faced = state['balls_faced']
    fours = state['fours']
    sixes = state['sixes']
    balls_per_boundary = (balls_faced) / (fours + sixes)
    return {"balls_per_boundary": balls_per_boundary}

def boundary_percentage(state: BatsmanState) -> BatsmanState:
    """Calculates boundary percentage from runs and balls faced"""
    runs = state['runs']
    four_runs = state['fours'] * 4
    six_runs = state['sixes'] * 6
    total_runs_in_boundaries = four_runs + six_runs
    boundary_percentage = (total_runs_in_boundaries / runs) * 100
    return {"boundary_percentage": boundary_percentage}

def summary(state: BatsmanState) -> BatsmanState:
    """Summarizes the batsman's performance"""
    runs = state['runs']
    balls_faced = state['balls_faced']
    fours = state['fours']
    sixes = state['sixes']
    strike_rate = state['strike_rate']
    balls_per_boundary = state['balls_per_boundary']
    boundary_percentage = state['boundary_percentage']
    state['summary'] = f"Batsman scored {runs} runs in {balls_faced} balls with a strike rate of {strike_rate}, balls per boundary of {balls_per_boundary}, and boundary percentage of {boundary_percentage}"
    return state

#define graph
batsman_graph = StateGraph(BatsmanState)

#add nodes to graph
batsman_graph.add_node("strike_rate", strike_rate)
batsman_graph.add_node("balls_per_boundary", balls_per_boundary)
batsman_graph.add_node("boundary_percentage", boundary_percentage)
batsman_graph.add_node("summary", summary)

#add edges to graph
batsman_graph.add_edge(START, "strike_rate")
batsman_graph.add_edge(START, "balls_per_boundary")
batsman_graph.add_edge(START, "boundary_percentage")

batsman_graph.add_edge("strike_rate", "summary")
batsman_graph.add_edge("balls_per_boundary", "summary")
batsman_graph.add_edge("boundary_percentage", "summary")

batsman_graph.add_edge("summary", END)

batsman_workflow = batsman_graph.compile()

initial_state = {"runs": 100, "balls_faced": 50, "fours": 10, "sixes": 4}
final_state = batsman_workflow.invoke(initial_state)
print("final_state: ", final_state)