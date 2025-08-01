import os
import sys
import json
import logging
import psutil
from datetime import datetime, timedelta
from dotenv import load_dotenv
import google.generativeai as genai

# --- Configuration & Setup ---

def load_configuration():
    """Loads configuration from config.json."""
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error("Error: config.json not found. Please copy config.json.example to config.json.")
        sys.exit(1)
    except json.JSONDecodeError:
        logging.error("Error: Could not decode config.json. Please check its format.")
        sys.exit(1)

def setup_logging(config):
    """Sets up logging based on the configuration."""
    log_config = config.get('logging', {})
    log_file = log_config.get('log_file', 'logs/optimizer.log')
    log_level_str = log_config.get('log_level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def configure_gemini():
    """Configures the Gemini API."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logging.error("Error: GEMINI_API_KEY not found in .env file.")
        sys.exit(1)
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        logging.error(f"Failed to configure Gemini: {e}")
        sys.exit(1)

# --- Scanning & Analysis ---

def get_partitions_to_scan(exclude_paths):
    """Gets all mounted disk partitions, filtering out system-critical and excluded paths."""
    partitions = []
    for part in psutil.disk_partitions(all=False):
        # Basic check for physical devices
        if 'loop' in part.device or 'snap' in part.device:
            continue
        
        # Check against exclude list
        is_excluded = any(part.mountpoint.startswith(p) for p in exclude_paths)
        if not is_excluded:
            partitions.append(part.mountpoint)
            logging.info(f"Found partition to scan: {part.mountpoint} ({part.fstype})")
        else:
            logging.warning(f"Excluding system partition: {part.mountpoint}")
            
    return partitions

def find_optimization_candidates(path, config):
    """Scans a directory and yields files/folders that are candidates for optimization."""
    # This is a placeholder for the core scanning logic
    # In the full implementation, this will walk the filesystem
    logging.info(f"Scanning {path} for optimization candidates...")
    # yield {"type": "large_file", "path": "/path/to/large/file.log", "size_mb": 1024}
    # yield {"type": "old_directory", "path": "/path/to/old_project", "last_modified": "2022-01-01"}
    pass # To be implemented

# --- AI Suggestions ---

def get_ai_suggestions(candidates):
    """Sends candidate list to Gemini and gets cleanup suggestions."""
    if not candidates:
        return []

    # This is a placeholder for the Gemini API call
    logging.info("Getting suggestions from Gemini AI...")
    
    # Placeholder response
    suggestions = [
        {"action": "delete", "path": "/path/to/large/file.log", "reason": "Large log file, likely safe to delete."},
        {"action": "archive", "path": "/path/to/old_project", "reason": "Old project, can be archived to save space."}
    ]
    return suggestions

# --- User Interaction & Execution ---

def confirm_and_execute(suggestions, dry_run=True):
    """Asks user to confirm suggestions and executes them."""
    if not suggestions:
        logging.info("No cleanup suggestions to execute.")
        return

    print("\n--- Optimization Suggestions ---")
    for i, sug in enumerate(suggestions):
        print(f"{i+1}. Action: {sug['action'].upper()}")
        print(f"   Path:   {sug['path']}")
        print(f"   Reason: {sug['reason']}")
        print("-" * 20)
    
    if dry_run:
        logging.info("Dry run is enabled. No changes will be made.")
        return

    try:
        response = input("Do you want to apply these changes? (yes/no): ").lower()
        if response != 'yes':
            print("Cleanup cancelled by user.")
            return
    except EOFError:
        # Running in a non-interactive environment (like cron)
        logging.warning("Non-interactive mode detected. Cannot ask for confirmation.")
        logging.warning("To run in cron, either enable dry_run or find a way to provide input.")
        return

    logging.info("Executing cleanup actions...")
    # Placeholder for execution logic
    # for sug in suggestions:
    #     if sug['action'] == 'delete':
    #         os.remove(sug['path']) # Add error handling
    #     elif sug['action'] == 'archive':
    #         # shutil.make_archive(...)
    #         pass
    print("Cleanup complete.")

# --- Main ---

def main():
    """Main function to run the volume optimizer."""
    config = load_configuration()
    setup_logging(config)
    # configure_gemini() # Uncomment when ready to use Gemini

    logging.info("--- Starting Volume Optimizer ---")
    
    scan_cfg = config.get('scan_options', {})
    safety_cfg = config.get('safety', {})
    
    partitions = get_partitions_to_scan(scan_cfg.get('exclude_paths', []))
    
    all_candidates = []
    for part in partitions:
        candidates = find_optimization_candidates(part, scan_cfg)
        # all_candidates.extend(list(candidates)) # Collect candidates

    # For demonstration, using dummy candidates
    all_candidates = [
        {"type": "large_file", "path": "/var/tmp/big_log_file.log", "size_mb": 1024, "age_days": 10},
        {"type": "old_directory", "path": "/home/user/projects/abandoned_proj", "size_mb": 250, "age_days": 400}
    ]

    suggestions = get_ai_suggestions(all_candidates)
    confirm_and_execute(suggestions, dry_run=safety_cfg.get('dry_run', True))
    
    logging.info("--- Volume Optimizer Finished ---")

if __name__ == '__main__':
    main()
