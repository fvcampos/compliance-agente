"""
Defines the state structures used in the LangGraph execution flow.
"""

from typing import List, TypedDict

class AgentState(TypedDict):
    """
    Represents the internal state of the Compliance Agent during a single
    execution run.

    Attributes:
        question (str): The incoming user query.
        generation (str): The current answer draft produced by the LLM.
        documents (List[str]): A list of context strings retrieved from the
                               vector database.
        retry_count (int): A counter to track how many times the agent has
                           tried to self-correct (to prevent infinite loops).
        grade (str): The relevance grade assigned to the retrieved documents
                     ("relevant" or "irrelevant").
    """
    question: str
    generation: str
    documents: List[str]
    retry_count: int
    grade: str
