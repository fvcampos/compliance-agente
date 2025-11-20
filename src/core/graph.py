"""
Defines the structure of the LangGraph application.
It maps the nodes (workers) to the edges (logic flow).
"""

from langgraph.graph import END, StateGraph, START

from src.core.state import AgentState
from src.agents.nodes import retrieve, grade_documents, generate

# 1. Initialize the Graph with our TypedDict State
workflow: StateGraph = StateGraph(AgentState)

# 2. Add the Nodes (The Workers)
# syntax: workflow.add_node("name_of_node", function_to_call)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)

# 3. Define the Edges (The Logic Flow)
# For this MVP step, we connect them linearly.
# Logic: Start -> Retrieve -> Grade -> Generate -> End
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_edge("grade_documents", "generate")
workflow.add_edge("generate", END)

# 4. Compile the Graph
# This creates the "Runnable" application that we can invoke.
app: StateGraph = workflow.compile()
