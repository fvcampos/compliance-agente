"""
Defines the structure of the LangGraph application.
It maps the nodes (workers) to the edges (logic flow).
"""

import logging

from langgraph.graph import END, StateGraph, START

from src.core.state import AgentState
from src.agents.nodes import (
    retrieve,
    grade_documents,
    generate,
    rewrite_query
)

def decide_to_generate(
        state: AgentState,
        max_retries: int = 3,
        default_generate_proceed: str = "yes") -> str:

    """
    Determines the next step in the graph execution flow based on the relevance
    grade of retrieved documents and the current retry count.

    This function acts as a conditional edge in the state graph. It checks if
    the retrieved documents are relevant ('yes') or if the maximum number of
    retries has been exceeded. If either condition is met, it directs the flow
    to the 'generate' node. Otherwise, it directs the flow to 'rewrite_query'
    to refine the search.

    Args:
        state (AgentState): The current state of the agent, containing keys
            like 'grade' (str) and 'retry_count' (int).
        max_retries (int, optional): The maximum number of times the query
            can be rewritten before forcing generation. Defaults to 3.
        default_generate_proceed (str, optional): The grade value that
            indicates documents are relevant enough to proceed. Defaults to
            "yes".

    Returns:
        str: The name of the next node to execute ("generate" or
        "rewrite_query").
    """

    logging.info("--- DECISION LOGIC ---")
    grade: str = state["grade"]
    retry_count: int = state.get("retry_count", 0)

    # Safety Valve: If we tried 3 times, just give up and generate 
    # (to prevent infinite loops / $$ costs)
    if retry_count >= max_retries:
        logging.info("--- DECISION: MAX RETRIES REACHED -> GENERATE ---")
        return "generate"

    if grade == default_generate_proceed:
        logging.info("--- DECISION: DOCS RELEVANT -> GENERATE ---")
        return "generate"
    else:
        logging.info("--- DECISION: DOCS IRRELEVANT -> REWRITE ---")
        return "rewrite_query"


# 1. Initialize the Graph with our TypedDict State
workflow: StateGraph = StateGraph(AgentState)

# 2. Add the Nodes (The Workers)
# syntax: workflow.add_node("name_of_node", function_to_call)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("rewrite_query", rewrite_query)

# 3. Define the Edges (The Logic Flow)
# For this MVP step, we connect them linearly.
# Logic: Start -> Retrieve -> Grade -> Generate -> End
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")

# Conditional Edge: Decide whether to Generate or Rewrite Query
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "generate",
        "rewrite_query": "rewrite_query"
    }
)

# Edge from Rewrite Query back to Retrieve
workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("generate", END)

# 4. Compile the Graph
# This creates the "Runnable" application that we can invoke.
app: StateGraph = workflow.compile()
