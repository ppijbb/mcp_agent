import os
import sys
import json
import logging
import psutil
import fnmatch
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

def is_admin():
    """Checks for administrator/root privileges across platforms."""
    try:
        # For Unix-like systems (Linux, macOS)
        return os.geteuid() == 0
    except AttributeError:
        # For Windows
        import ctypes
        return ctypes.windll.shell32.IsUserAnAdmin() != 0

# --- Scanning & Analysis ---

def get_partitions_to_scan(exclude_paths):
    """Gets all mounted disk partitions, filtering out system-critical and excluded paths."""
    partitions = []
    logging.info("Searching for partitions to scan...")
    for part in psutil.disk_partitions(all=False):
        # Filter out virtual, temporary, or special filesystems
        if 'loop' in part.device or 'snap' in part.device or 'tmpfs' in part.fstype:
            continue
        
        # Check against a more robust exclude list
        is_excluded = any(part.mountpoint.startswith(p) for p in exclude_paths if p != "C:\\Windows" or os.name == 'nt')
        if not is_excluded:
            partitions.append(part.mountpoint)
            logging.info(f"Found partition to scan: {part.mountpoint} ({part.fstype})")
        else:
            logging.warning(f"Excluding configured partition: {part.mountpoint}")
            
    return partitions

def find_optimization_candidates(path, config):
    """Scans a directory and yields files/folders that are candidates for optimization."""
    candidates = []
    inaccessible_paths = []
    
    large_file_mb = config.get('large_file_threshold_mb', 500)
    old_file_days = config.get('old_file_threshold_days', 90)
    temp_patterns = config.get('temp_patterns', [])
    cache_dirs = config.get('cache_dirs', [])
    exclude_paths = config.get('exclude_paths', [])
    
    now = datetime.now()
    large_file_bytes = large_file_mb * 1024 * 1024
    old_file_delta = timedelta(days=old_file_days)

    logging.info(f"Starting scan on: {path}")
    
    for root, dirs, files in os.walk(path, topdown=True, onerror=lambda e: inaccessible_paths.append(e.filename)):
        try:
            # Filter out excluded directories to prevent descending into them
            dirs[:] = [d for d in dirs if not any(os.path.join(root, d).startswith(p) for p in exclude_paths)]
            
            # Identify cache directories
            for d in list(dirs):
                if d in cache_dirs:
                    dir_path = os.path.join(root, d)
                    candidates.append({"type": "cache_dir", "path": dir_path, "reason": f"Matches cache pattern: {d}"})
                    dirs.remove(d) # Don't descend into cache dirs
            
            # Scan files
            for name in files:
                file_path = os.path.join(root, name)
                
                # Check for temp files by pattern
                if any(fnmatch.fnmatch(name.lower(), pat) for pat in temp_patterns):
                    candidates.append({"type": "temp_file", "path": file_path, "reason": f"Matches temp pattern: {name}"})
                    continue # Don't double-count

                stat = os.stat(file_path)
                mtime = datetime.fromtimestamp(stat.st_mtime)
                size_bytes = stat.st_size
                
                # Check for large files
                if size_bytes > large_file_bytes:
                    candidates.append({
                        "type": "large_file",
                        "path": file_path,
                        "size_mb": round(size_bytes / (1024 * 1024), 2),
                        "reason": f"Larger than {large_file_mb} MB"
                    })

                # Check for old files
                if now - mtime > old_file_delta:
                     candidates.append({
                        "type": "old_file",
                        "path": file_path,
                        "last_modified": mtime.strftime("%Y-%m-%d"),
                        "reason": f"Older than {old_file_days} days"
                    })

        except PermissionError:
            inaccessible_paths.append(root)
            continue
        except FileNotFoundError:
            # File might have been deleted during the scan
            continue
            
    return candidates, list(set(inaccessible_paths))

# --- AI Suggestions ---

def get_ai_suggestions(candidates):
    """Sends candidate list to Gemini and gets cleanup suggestions."""
    if not candidates:
        return []
    # This is a placeholder for the Gemini API call
    logging.info("Getting suggestions from Gemini AI...")
    # This will be replaced with actual AI logic
    return candidates # For now, just return the candidates as suggestions

# --- User Interaction & Execution ---

def confirm_and_execute(suggestions, dry_run=True):
    """Asks user to confirm suggestions and executes them."""
    if not suggestions:
        logging.info("No cleanup suggestions to execute.")
        return

    print("\n--- Optimization Suggestions ---")
    for i, sug in enumerate(suggestions):
        print(f"{i+1}. Type: {sug['type'].replace('_', ' ').title()}")
        print(f"   Path:   {sug['path']}")
        print(f"   Reason: {sug['reason']}")
        print("-" * 20)
    
    if dry_run:
        logging.info("Dry run is enabled. No changes will be made.")
        return

    try:
        response = input("Do you want to apply these changes? (yes/no): ").lower().strip()
        if response != 'yes':
            print("Cleanup cancelled by user.")
            return
    except (EOFError, KeyboardInterrupt):
        logging.warning("\nNon-interactive mode or user interruption. Cancelling cleanup.")
        return

    logging.info("Executing cleanup actions...")
    # Placeholder for execution logic
    print("Cleanup would be performed here.")
    logging.info("Cleanup complete.")

# --- Main ---

def main():
    """Main function to run the volume optimizer."""
    config = load_configuration()
    setup_logging(config)
    # configure_gemini() # Uncomment when ready to use Gemini

    logging.info("--- Starting Volume Optimizer ---")

    if is_admin():
        logging.warning("="*60)
        logging.warning("WARNING: Running with root/administrator privileges!")
        logging.warning("This allows access to all files. Review suggestions carefully.")
        logging.warning("="*60)
    
    scan_cfg = config.get('scan_options', {})
    safety_cfg = config.get('safety', {})
    
    partitions = get_partitions_to_scan(scan_cfg.get('exclude_paths', []))
    
    all_candidates = []
    all_inaccessible = []
    for part in partitions:
        candidates, inaccessible = find_optimization_candidates(part, scan_cfg)
        all_candidates.extend(candidates)
        all_inaccessible.extend(inaccessible)

    if all_inaccessible:
        logging.warning("\n--- Inaccessible Paths ---")
        logging.warning("Could not scan the following paths due to permission errors:")
        for path in sorted(list(set(all_inaccessible)))[:15]: # Show a sample
             logging.warning(f" - {path}")
        if len(all_inaccessible) > 15:
            logging.warning(f" ... and {len(all_inaccessible) - 15} more.")
        logging.warning("To include these paths, you may need to run this script with administrator/sudo privileges.")

    suggestions = get_ai_suggestions(all_candidates)
    confirm_and_execute(suggestions, dry_run=safety_cfg.get('dry_run', True))
    
    logging.info("--- Volume Optimizer Finished ---")

if __name__ == '__main__':
    main()
