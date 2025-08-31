import uuid
import logging
from .graph import AnnWorkflow

def main():
    """
    Main function to run the ANN-inspired workflow.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Unique ID for the run
        run_id = f"ann_run_{uuid.uuid4().hex[:8]}"
        
        # Initialize the workflow
        workflow = AnnWorkflow(max_revisions=5)
        
        # Define the initial task
        initial_task = "Create a Python script that calculates the factorial of a number with proper error handling and input validation."
        
        # Initial state
        initial_state = {
            "initial_task": initial_task,
        }
        
        logger.info(f"🚀 Starting ANN Workflow with run_id: {run_id}")
        logger.info(f"📝 Initial Task: {initial_task}")
        logger.info("=" * 60)
        
        # Run the workflow
        final_state = workflow.run(initial_state)
        
        # Display results
        logger.info("=" * 60)
        logger.info("🏁 Workflow completed!")
        
        if final_state.get("error"):
            logger.error(f"❌ Workflow failed: {final_state['error']}")
            logger.error(f"📊 Final Status: {final_state.get('status', 'unknown')}")
        else:
            logger.info(f"✅ Workflow completed successfully!")
            logger.info(f"📊 Final Status: {final_state.get('status', 'unknown')}")
            logger.info(f"🔄 Total Revisions: {final_state.get('revision_number', 0)}")
            
            if final_state.get("code"):
                logger.info("📝 Generated Code:")
                logger.info("-" * 40)
                logger.info(final_state["code"])
                logger.info("-" * 40)
            
            if final_state.get("execution_result"):
                logger.info("🔍 Execution Result:")
                logger.info("-" * 40)
                logger.info(final_state["execution_result"])
                logger.info("-" * 40)
        
        return final_state
        
    except Exception as e:
        logger.error(f"❌ Fatal error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
