"""
Entry point for running the agent from the command line.
"""

import logging

from src.core.graph import app

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

def run_agent():
    """
    Runs the Compliance Agent with a sample query.
    """
    logging.info("Starting Compliance Agent...")

    # 1. Define the Input Question
    # (Make sure this relates to the PDF you ingested!)
    user_question = "What does it say about work flexibility?"

    initial_state: dict = {
        "question": user_question,
        "generation": "",
        "documents": [],
        "retry_count": 0
    }
    
    # 2. Invoke the Graph
    # The 'app.invoke' method passes the state through the nodes defined in
    # graph.py
    logging.info(f"User Question: {user_question}")
    
    try:
        final_state = app.invoke(initial_state)
        
        # 3. Print the Result
        logging.info("--- FINAL ANSWER ---")
        logging.info(final_state["generation"])
        logging.info("-----------------------")
        
    except Exception as e:
        logger.error(f"Agent execution failed: {e}", exc_info=True)

if __name__ == "__main__":
    run_agent()
