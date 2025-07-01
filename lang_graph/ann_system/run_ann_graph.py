import uuid
from .graph import AnnWorkflow

def main():
    """
    Main function to run the ANN-inspired workflow.
    """
    # Unique ID for the run
    run_id = f"ann_run_{uuid.uuid4().hex[:4]}"

    # Initialize the workflow
    workflow = AnnWorkflow()

    # Define the initial task
    initial_task = "Create a python script that calculates the factorial of a number."

    # Initial state
    initial_state = {
        "initial_task": initial_task,
        "plan": None,
        "code": None,
        "critique": None,
        "execution_result": None,
        "history": [],
        "revision_number": 0,
    }

    print(f"🚀 Starting ANN Workflow with run_id: {run_id}")
    print(f"📝 Initial Task: {initial_task}\n")

    # Run the graph
    # Since we haven't defined edges yet, this will only run the entry point node ("planner").
    for event in workflow.app.stream(initial_state, {"recursion_limit": 10}, stream_mode="values"):
        print("---STREAM EVENT---")
        print(event)
        print("\n")

    print("🏁 Workflow finished.")

if __name__ == "__main__":
    main()
