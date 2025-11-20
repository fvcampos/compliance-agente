"""
Defines the nodes (workers) for the LangGraph agent.
"""

import logging
import os
from typing import Dict, Any

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
    print("--- NODE: RETRIEVE ---")
    question = state["question"]

    # Pass arguments as a dictionary!
    documents_str: str = retrieve_documents.invoke({
        "query": question, 
        "chunk_limit": 3
    })

    return {"documents": [documents_str]}


def grade_documents(state: AgentState) -> Dict[str, Any]:
    """Node 2: The Compliance Officer (Gemini)"""
    print("--- NODE: GRADE DOCUMENTS ---")
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

    print(f"--- JUDGE DECISION: {score.binary_score} ---")
    return {"question": question, "documents": state["documents"]}


def generate(state: AgentState) -> Dict[str, Any]:
    """Node 3: The Writer (Gemini)"""
    print("--- NODE: GENERATE ---")
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
