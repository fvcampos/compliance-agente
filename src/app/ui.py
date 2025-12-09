"""
The Chat Interface using Chainlit.
Run this with: uv run chainlit run src/app/ui.py -w
"""

import os

import chainlit as cl
from src.core.graph import app
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

collector_endpoint = os.getenv(
    "PHOENIX_COLLECTOR_ENDPOINT", "http://127.0.0.1:6006/v1/traces")

tracer_provider = register(
    project_name="compliance-agent", 
    endpoint=collector_endpoint
)

LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

@cl.on_chat_start
async def start():
    """
    Initializes the chat session.
    """
    # Send a welcome message
    await cl.Message(
        content=(
            "**Compliance Agent Online.**\nI can check company policies for "
            "you. Try asking about 'Remote Work' or 'Spending Limits'."
        )
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """
    The main chat loop.
    """
    # 1. Initialize the State for this specific run
    initial_state = {
        "question": message.content,
        "generation": "",
        "documents": [],
        "retry_count": 0,
        "grade": "",
    }

    # 2. Setup the "Thinking" UI (Steps)
    # We create a final message placeholder but don't send it yet
    final_answer = cl.Message(content="")

    # 3. Stream the Graph Execution
    try:
        # 'astream' yields the output of each node as it finishes
        async for output in app.astream(initial_state):
            for node_name, node_output in output.items():

                # --- VISUALIZE THE STEPS ---

                if node_name == "retrieve":
                    async with cl.Step(name="Retriever", type="tool") as step:
                        step.input = "Searching Vector DB..."
                        step.output = "Found relevant chunks."

                elif node_name == "grade_documents":
                    grade = node_output.get("grade", "unknown")
                    async with cl.Step(name="Auditor", type="llm") as step:
                        if grade == "yes":
                            step.output = "Documents are relevant."
                        else:
                            step.output = (
                                "Documents are irrelevant. Requesting query "
                                "rewrite."
                            )

                elif node_name == "rewrite_query":
                    new_q = node_output.get("question", "")
                    async with cl.Step(
                        name="Query Refiner", type="run"
                    ) as step:
                        step.output = f"New Query: '{new_q}'"

                elif node_name == "generate":
                    answer_text = node_output.get("generation", "")
                    final_answer.content = answer_text

        # 4. Send final answer only if we succeeded without error
        if final_answer.content:
            await final_answer.send()
        else:
            await cl.Message(content="Unable to generate an answer.").send()

    except Exception as e:
        error_msg = f"**System Error:** {str(e)}"
        if "429" in str(e) or "ResourceExhausted" in str(e):
            error_msg = (
                "**Rate Limit Hit:** The free AI quota is exhausted. Please "
                "wait a moment and try again."
            )

        await cl.Message(content=error_msg).send()
