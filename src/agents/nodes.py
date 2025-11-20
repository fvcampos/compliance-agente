"""
Defines the nodes (workers) for the LangGraph agent.
"""

import logging
from typing import Dict, Any, List, Tuple

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from src.core.state import AgentState
from src.utils.settings import settings
from src.agents.tools import retrieve_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    max_retries=2,
    google_api_key=settings.GOOGLE_API_KEY
)

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


def retrieve(state: AgentState) -> Dict[str, Any]:
    """Node 1: The Researcher"""
    logging.info("--- NODE: RETRIEVE ---")
    question: str = state["question"]

    # Pass arguments as a dictionary!
    documents_str: str = retrieve_documents.invoke({
        "query": question, 
        "chunk_limit": 3
    })

    return {"documents": [documents_str]}


def grade_documents(state: AgentState) -> Dict[str, Any]:
    """Node 2: The Compliance Officer (Gemini)"""
    logging.info("--- NODE: GRADE DOCUMENTS ---")
    question: str = state["question"]
    documents: str = state["documents"][0]

    # Gemini supports structured output too!
    structured_llm_grader: ChatGoogleGenerativeAI = llm.with_structured_output(
        GradeDocuments)

    system: str = """You are a strict compliance auditor assessing relevance. 
    If the document contains keyword(s) or semantic meaning related to the user
    question, grade it as relevant. Give a binary score 'yes' or 'no'."""

    grade_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User " +
             "question: {question}"),
        ]
    )

    retrieval_grader: ChatGoogleGenerativeAI = grade_prompt | \
        structured_llm_grader

    score: GradeDocuments = retrieval_grader.invoke(
        {"question": question,
         "document": documents}
    )

    logging.info(f"--- JUDGE DECISION: {score.binary_score} ---")
    return {"question": question, "documents": state["documents"]}


def generate(state: AgentState) -> Dict[str, Any]:
    """Node 3: The Writer (Gemini)"""
    logging.info("--- NODE: GENERATE ---")
    question: str = state["question"]
    documents: str = state["documents"][0]

    prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(
        """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Keep the answer concise.
        
        Question: {question} 
        Context: {documents} 
        
        Answer:"""
    )

    rag_chain: ChatGoogleGenerativeAI = prompt | llm
    response: ChatGoogleGenerativeAI = rag_chain.invoke(
        {"documents": documents,
         "question": question}
    )

    return {"generation": response.content}


def rewrite_query(state: AgentState) -> Dict[str, Any]:
    '''

    This node is triggered when the retrieved documents are graded as irrelevant
    to the user's question. It utilizes an LLM to transform the original
    question into a more effective search query optimized for vector retrieval,
    aiming to capture the semantic meaning better than the initial attempt.
    Args:
        state (AgentState): The current state of the agent graph, containing 
                            the original 'question' and the current
                            'retry_count'.
    Returns:
        Dict[str, Any]: A dictionary containing:
            - "question" (str): The newly rewritten, optimized question string.
            - "retry_count" (int): The incremented retry counter to track the
            number of refinement attempts.
    
    '''
    
    logging.info("--- NODE: REWRITE QUERY ---")
    question: str = state["question"]

    # A specific prompt to act as a "Translator"
    # "Look at the initial question and formulate an improved question 
    # that is more likely to retrieve relevant facts."
    msg: List[Tuple[str, str]] = [
        ("system", "You are a query rewriter that converts an input question "
                   "to a better version that is optimized for vector "
                   "retrieval. Look at the initial and formulate an improved "
                   "question."),
        ("human", f"Initial Question: {question} \n Formulate an improved "
                  "question."),
    ]

    better_question = llm.invoke(msg)
    
    logging.info(f"--- REWRITTEN QUERY: {better_question.content} ---")
    
    # Update the state with the NEW question
    # Also increment the retry counter to prevent infinite loops later
    return {
        "question": better_question.content,
        "retry_count": state.get("retry_count", 0) + 1
    }
