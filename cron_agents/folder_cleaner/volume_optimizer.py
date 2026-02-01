import os
import sys
import json
import logging
import psutil
import fnmatch
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
import google.generativeai as genai

# --- Data Models ---


@dataclass
class CleanupCandidate:
    """Represents a file or directory candidate for cleanup."""
    type: str
    path: str
    reason: str
    size_mb: Optional[float] = None
    last_modified: Optional[str] = None
    confidence: float = 1.0


@dataclass
class CleanupAction:
    """Represents a cleanup action to be performed."""
    action_type: str  # 'delete', 'compress', 'move'
    target_path: str
    reason: str
    estimated_space_saved_mb: float
    risk_level: str  # 'low', 'medium', 'high'

# --- Configuration & Setup ---


def load_configuration() -> Dict[str, Any]:
    """Loads configuration from config.json with validation."""
    config_path = Path('config.json')

    if not config_path.exists():
        raise FileNotFoundError("config.json not found. Please copy config.json.example to config.json.")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Validate required sections
        required_sections = ['scan_options', 'safety', 'logging']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")

        return config

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config.json: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration: {e}")


def setup_logging(config: Dict[str, Any]) -> None:
    """Sets up logging based on the configuration."""
    log_config = config.get('logging', {})
    log_file = log_config.get('log_file', 'logs/optimizer.log')
    log_level_str = log_config.get('log_level', 'INFO').upper()

    try:
        log_level = getattr(logging, log_level_str, logging.INFO)
    except AttributeError:
        log_level = logging.INFO

    # Create log directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Clear existing handlers to avoid duplicates
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def configure_gemini() -> genai.GenerativeModel:
    """Configures the Gemini API and returns the model."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to configure Gemini: {e}")


def is_admin() -> bool:
    """Checks for administrator/root privileges across platforms."""
    try:
        # For Unix-like systems (Linux, macOS)
        return os.geteuid() == 0
    except AttributeError:
        # For Windows
        try:
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except Exception:
            return False

# --- Scanning & Analysis ---


def get_partitions_to_scan(exclude_paths: List[str]) -> List[str]:
    """Gets all mounted disk partitions, filtering out system-critical and excluded paths."""
    partitions = []
    logger = logging.getLogger(__name__)

    logger.info("Searching for partitions to scan...")

    try:
        for part in psutil.disk_partitions(all=False):
            # Filter out virtual, temporary, or special filesystems
            if any(keyword in part.device.lower() for keyword in ['loop', 'snap', 'tmpfs', 'proc', 'sysfs']):
                continue

            # Check against exclude list
            is_excluded = False
            for exclude_path in exclude_paths:
                if part.mountpoint.startswith(exclude_path):
                    # Special handling for Windows
                    if exclude_path == "C:\\Windows" and os.name == 'nt':
                        is_excluded = True
                    elif exclude_path != "C:\\Windows":
                        is_excluded = True
                    break

            if not is_excluded:
                partitions.append(part.mountpoint)
                logger.info(f"Found partition to scan: {part.mountpoint} ({part.fstype})")
            else:
                logger.debug(f"Excluding configured partition: {part.mountpoint}")

    except Exception as e:
        logger.error(f"Error scanning partitions: {e}")
        raise RuntimeError(f"Failed to scan partitions: {e}")

    return partitions


def find_optimization_candidates(path: str, config: Dict[str, Any]) -> Tuple[List[CleanupCandidate], List[str]]:
    """Scans a directory and returns files/folders that are candidates for optimization."""
    candidates = []
    inaccessible_paths = []
    logger = logging.getLogger(__name__)

    scan_options = config.get('scan_options', {})
    large_file_mb = scan_options.get('large_file_threshold_mb', 500)
    old_file_days = scan_options.get('old_file_threshold_days', 90)
    temp_patterns = scan_options.get('temp_patterns', [])
    cache_dirs = scan_options.get('cache_dirs', [])
    exclude_paths = scan_options.get('exclude_paths', [])

    now = datetime.now()
    large_file_bytes = large_file_mb * 1024 * 1024
    old_file_delta = timedelta(days=old_file_days)

    logger.info(f"Starting scan on: {path}")

    def handle_error(error):
        inaccessible_paths.append(error.filename)
        logger.debug(f"Permission denied: {error.filename}")

    try:
        for root, dirs, files in os.walk(path, topdown=True, onerror=handle_error):
            try:
                # Filter out excluded directories
                dirs[:] = [d for d in dirs if not any(
                    os.path.join(root, d).startswith(p) for p in exclude_paths
                )]

                # Identify cache directories
                for d in list(dirs):
                    if d in cache_dirs:
                        dir_path = os.path.join(root, d)
                        try:
                            size_bytes = get_directory_size(dir_path)
                            candidates.append(CleanupCandidate(
                                type="cache_dir",
                                path=dir_path,
                                reason=f"Matches cache pattern: {d}",
                                size_mb=round(size_bytes / (1024 * 1024), 2)
                            ))
                            dirs.remove(d)  # Don't descend into cache dirs
                        except (PermissionError, OSError):
                            continue

                # Scan files
                for name in files:
                    file_path = os.path.join(root, name)

                    try:
                        stat = os.stat(file_path)
                        mtime = datetime.fromtimestamp(stat.st_mtime)
                        size_bytes = stat.st_size

                        # Check for temp files by pattern
                        if any(fnmatch.fnmatch(name.lower(), pat.lower()) for pat in temp_patterns):
                            candidates.append(CleanupCandidate(
                                type="temp_file",
                                path=file_path,
                                reason=f"Matches temp pattern: {name}",
                                size_mb=round(size_bytes / (1024 * 1024), 2)
                            ))
                            continue

                        # Check for large files
                        if size_bytes > large_file_bytes:
                            candidates.append(CleanupCandidate(
                                type="large_file",
                                path=file_path,
                                reason=f"Larger than {large_file_mb} MB",
                                size_mb=round(size_bytes / (1024 * 1024), 2)
                            ))

                        # Check for old files
                        if now - mtime > old_file_delta:
                            candidates.append(CleanupCandidate(
                                type="old_file",
                                path=file_path,
                                reason=f"Older than {old_file_days} days",
                                last_modified=mtime.strftime("%Y-%m-%d"),
                                size_mb=round(size_bytes / (1024 * 1024), 2)
                            ))

                    except (PermissionError, OSError, FileNotFoundError):
                        continue

            except (PermissionError, OSError):
                inaccessible_paths.append(root)
                continue

    except Exception as e:
        logger.error(f"Error scanning {path}: {e}")
        raise RuntimeError(f"Failed to scan directory {path}: {e}")

    return candidates, list(set(inaccessible_paths))


def get_directory_size(path: str) -> int:
    """Calculate the total size of a directory."""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    continue
    except (OSError, PermissionError):
        pass
    return total_size

# --- AI Suggestions ---


def get_ai_suggestions(candidates: List[CleanupCandidate], model: genai.GenerativeModel) -> List[CleanupAction]:
    """Sends candidate list to Gemini and gets intelligent cleanup suggestions."""
    if not candidates:
        return []

    logger = logging.getLogger(__name__)
    logger.info("Getting suggestions from Gemini AI...")

    try:
        # Prepare data for AI analysis
        candidates_data = []
        for candidate in candidates:
            candidates_data.append({
                "type": candidate.type,
                "path": candidate.path,
                "reason": candidate.reason,
                "size_mb": candidate.size_mb,
                "last_modified": candidate.last_modified
            })

        # Create prompt for Gemini
        prompt = f"""
        Analyze the following file and directory cleanup candidates and provide intelligent suggestions.
        For each candidate, suggest the most appropriate action (delete, compress, move) with risk assessment.

        Candidates:
        {json.dumps(candidates_data, indent=2)}

        Please respond with a JSON array of cleanup actions. Each action should have:
        - action_type: "delete", "compress", or "move"
        - target_path: the file/directory path
        - reason: why this action is recommended
        - estimated_space_saved_mb: estimated space that will be saved
        - risk_level: "low", "medium", or "high"

        Focus on safety and only recommend deletion for clearly safe items like temp files and cache directories.
        """

        response = model.generate_content(prompt)

        # Parse AI response
        try:
            suggestions_data = json.loads(response.text)
            actions = []

            for item in suggestions_data:
                actions.append(CleanupAction(
                    action_type=item.get('action_type', 'delete'),
                    target_path=item.get('target_path', ''),
                    reason=item.get('reason', ''),
                    estimated_space_saved_mb=item.get('estimated_space_saved_mb', 0.0),
                    risk_level=item.get('risk_level', 'medium')
                ))

            logger.info(f"AI generated {len(actions)} cleanup suggestions")
            return actions

        except json.JSONDecodeError:
            logger.warning("Failed to parse AI response as JSON, using fallback logic")
            return create_fallback_suggestions(candidates)

    except Exception as e:
        logger.error(f"Error getting AI suggestions: {e}")
        return create_fallback_suggestions(candidates)


def create_fallback_suggestions(candidates: List[CleanupCandidate]) -> List[CleanupAction]:
    """Create basic cleanup suggestions when AI is not available."""
    actions = []

    for candidate in candidates:
        if candidate.type in ['temp_file', 'cache_dir']:
            actions.append(CleanupAction(
                action_type='delete',
                target_path=candidate.path,
                reason=candidate.reason,
                estimated_space_saved_mb=candidate.size_mb or 0.0,
                risk_level='low'
            ))
        elif candidate.type == 'large_file':
            actions.append(CleanupAction(
                action_type='compress',
                target_path=candidate.path,
                reason=f"Large file: {candidate.reason}",
                estimated_space_saved_mb=(candidate.size_mb or 0.0) * 0.3,  # Estimate 30% compression
                risk_level='medium'
            ))
        elif candidate.type == 'old_file':
            actions.append(CleanupAction(
                action_type='move',
                target_path=candidate.path,
                reason=f"Old file: {candidate.reason}",
                estimated_space_saved_mb=candidate.size_mb or 0.0,
                risk_level='high'
            ))

    return actions

# --- User Interaction & Execution ---


def confirm_and_execute(actions: List[CleanupAction], dry_run: bool = True) -> None:
    """Asks user to confirm suggestions and executes them."""
    if not actions:
        logging.info("No cleanup suggestions to execute.")
        return

    logger = logging.getLogger(__name__)

    print("\n" + "="*60)
    print("CLEANUP SUGGESTIONS")
    print("="*60)

    total_space_saved = 0.0
    for i, action in enumerate(actions):
        print(f"\n{i+1}. Action: {action.action_type.upper()}")
        print(f"   Target: {action.target_path}")
        print(f"   Reason: {action.reason}")
        print(f"   Space Saved: {action.estimated_space_saved_mb:.2f} MB")
        print(f"   Risk Level: {action.risk_level.upper()}")
        total_space_saved += action.estimated_space_saved_mb
        print("-" * 40)

    print(f"\nTotal estimated space to be saved: {total_space_saved:.2f} MB")

    if dry_run:
        logger.info("Dry run is enabled. No changes will be made.")
        print("\n[DRY RUN] No actual changes will be made.")
        return

    try:
        print("\nDo you want to apply these changes?")
        response = input("Type 'yes' to proceed, 'no' to cancel: ").lower().strip()

        if response != 'yes':
            print("Cleanup cancelled by user.")
            return

    except (EOFError, KeyboardInterrupt):
        logger.warning("Non-interactive mode or user interruption. Cancelling cleanup.")
        print("\nCleanup cancelled.")
        return

    logger.info("Executing cleanup actions...")
    execute_cleanup_actions(actions)
    logger.info("Cleanup complete.")


def execute_cleanup_actions(actions: List[CleanupAction]) -> None:
    """Execute the approved cleanup actions."""
    logger = logging.getLogger(__name__)
    successful_actions = 0
    failed_actions = 0

    for action in actions:
        try:
            if action.action_type == 'delete':
                execute_delete_action(action)
            elif action.action_type == 'compress':
                execute_compress_action(action)
            elif action.action_type == 'move':
                execute_move_action(action)
            else:
                logger.warning(f"Unknown action type: {action.action_type}")
                continue

            successful_actions += 1
            logger.info(f"Successfully executed {action.action_type} on {action.target_path}")

        except Exception as e:
            failed_actions += 1
            logger.error(f"Failed to execute {action.action_type} on {action.target_path}: {e}")

    print(f"\nCleanup Summary:")
    print(f"  Successful: {successful_actions}")
    print(f"  Failed: {failed_actions}")


def execute_delete_action(action: CleanupAction) -> None:
    """Execute a delete action."""
    path = Path(action.target_path)

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {action.target_path}")

    if path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    else:
        raise ValueError(f"Unknown file type: {action.target_path}")


def execute_compress_action(action: CleanupAction) -> None:
    """Execute a compress action."""
    path = Path(action.target_path)

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {action.target_path}")

    if path.is_file():
        # Simple compression using gzip
        import gzip
        with open(path, 'rb') as f_in:
            with gzip.open(f"{path}.gz", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        path.unlink()  # Remove original file
    else:
        # For directories, create a tar.gz archive
        import tarfile
        with tarfile.open(f"{path}.tar.gz", "w:gz") as tar:
            tar.add(path, arcname=path.name)
        shutil.rmtree(path)  # Remove original directory


def execute_move_action(action: CleanupAction) -> None:
    """Execute a move action."""
    path = Path(action.target_path)

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {action.target_path}")

    # Create a backup directory
    backup_dir = Path.home() / "cleanup_backup"
    backup_dir.mkdir(exist_ok=True)

    # Move to backup directory
    destination = backup_dir / path.name
    shutil.move(str(path), str(destination))

# --- Main ---


def main() -> None:
    """Main function to run the volume optimizer."""
    try:
        # Load configuration
        config = load_configuration()
        setup_logging(config)

        logger = logging.getLogger(__name__)
        logger.info("--- Starting Volume Optimizer ---")

        # Check for admin privileges
        if is_admin():
            logger.warning("="*60)
            logger.warning("WARNING: Running with root/administrator privileges!")
            logger.warning("This allows access to all files. Review suggestions carefully.")
            logger.warning("="*60)

        # Get configuration sections
        scan_cfg = config.get('scan_options', {})
        safety_cfg = config.get('safety', {})

        # Scan partitions
        logger.info("Scanning partitions...")
        partitions = get_partitions_to_scan(scan_cfg.get('exclude_paths', []))

        if not partitions:
            logger.warning("No partitions found to scan.")
            return

        # Find optimization candidates
        logger.info("Finding optimization candidates...")
        all_candidates = []
        all_inaccessible = []

        for partition in partitions:
            try:
                candidates, inaccessible = find_optimization_candidates(partition, scan_cfg)
                all_candidates.extend(candidates)
                all_inaccessible.extend(inaccessible)
            except Exception as e:
                logger.error(f"Failed to scan partition {partition}: {e}")
                continue

        # Report inaccessible paths
        if all_inaccessible:
            logger.warning("\n--- Inaccessible Paths ---")
            logger.warning("Could not scan the following paths due to permission errors:")
            for path in sorted(list(set(all_inaccessible)))[:15]:
                logger.warning(f" - {path}")
            if len(all_inaccessible) > 15:
                logger.warning(f" ... and {len(all_inaccessible) - 15} more.")
            logger.warning("To include these paths, you may need to run this script with administrator/sudo privileges.")

        # Generate AI suggestions
        if all_candidates:
            try:
                model = configure_gemini()
                suggestions = get_ai_suggestions(all_candidates, model)
            except Exception as e:
                logger.error(f"Failed to get AI suggestions: {e}")
                logger.info("Using fallback suggestions...")
                suggestions = create_fallback_suggestions(all_candidates)
        else:
            logger.info("No optimization candidates found.")
            suggestions = []

        # Execute suggestions
        dry_run = safety_cfg.get('dry_run', True)
        confirm_and_execute(suggestions, dry_run=dry_run)

        logger.info("--- Volume Optimizer Finished ---")

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Volume optimizer failed: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
