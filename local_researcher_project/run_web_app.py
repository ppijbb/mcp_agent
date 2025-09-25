#!/usr/bin/env python3
"""
Run Local Researcher Web Application

This script starts the Streamlit web interface for the Local Researcher project.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Run the Streamlit web application."""
    try:
        # Get the project root directory
        project_root = Path(__file__).parent
        streamlit_app_path = project_root / "src" / "web" / "streamlit_app.py"
        
        # Check if the Streamlit app exists
        if not streamlit_app_path.exists():
            print(f"Error: Streamlit app not found at {streamlit_app_path}")
            sys.exit(1)
        
        # Run Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(streamlit_app_path),
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--browser.gatherUsageStats", "false"
        ]
        
        print("Starting Local Researcher Web Application...")
        print(f"App will be available at: http://localhost:8501")
        print("Press Ctrl+C to stop the application")
        
        subprocess.run(cmd, cwd=str(project_root))
        
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Error running web application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
