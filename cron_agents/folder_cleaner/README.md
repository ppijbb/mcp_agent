"""
# Volume Optimizer

A cron-based agent that scans all volumes, identifies optimization opportunities,
and asks for user consent before taking any action.

## Features

- Cross-platform: Works on Linux, macOS, and Windows.
- Scans all mounted partitions for potential cleanup.
- Identifies large files, old files, temporary files, and cache directories.
- Uses Gemini AI to intelligently suggest cleanup actions.
- **Safe by design:** No files are deleted without explicit user approval.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd folder_cleaner
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure the agent:**
    -   Copy the example config file:
        ```bash
        cp config.json.example config.json
        ```
    -   Edit `config.json` to fit your needs. You can specify thresholds for file sizes, age, and directories to exclude.

4.  **Set up environment variables:**
    -   Copy the example `.env` file:
        ```bash
        cp .env.example .env
        ```
    -   Open the `.env` file and add your Gemini API key:
        ```
        GEMINI_API_KEY=your_gemini_api_key_here
        ```

## How to Run

You can run the script manually for the first time to see how it works.

```bash
python volume_optimizer.py
```

This will perform a scan and print the optimization suggestions to the console, asking for your approval.

## Automating with Cron

To run the optimizer automatically, you can set up a cron job.

1.  Open your crontab editor:
    ```bash
    crontab -e
    ```

2.  Add a line to run the script at your desired interval. For example, to run it every Sunday at 3 AM:
    ```cron
    0 3 * * 0 /usr/bin/python3 /path/to/your/project/folder_cleaner/volume_optimizer.py >> /path/to/your/project/folder_cleaner/logs/cron.log 2>&1
    ```
    **Important:** Make sure to use the absolute paths to your python interpreter and the script.

## ⚠️ Disclaimer

This script is designed to be safe by requiring user confirmation. However, deleting files is an inherently risky operation. Always review the suggestions carefully before approving. The author is not responsible for any data loss.
"""