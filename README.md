# MCP - Agent

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure API keys:
   - Create `mcp_agent.secrets.yaml` file in the `srcs` directory
   - Add your API keys for OpenAI and Google:
     ```yaml
     openai:
       api_key: your-openai-api-key
     google:
       api_key: your-google-api-key
     ```

## Running the Agent

1. Navigate to the `srcs` directory:
   ```bash
   cd srcs
   ```

2. Run the researcher agent:
   ```bash
   python researcher.py
   ```

The agent will:
- Create an `agent_folder` directory for file operations
- Initialize connections to required MCP servers
- Execute the research task using OpenAI's GPT model
- Save the final report in markdown format

Note: Make sure you have Docker installed for the Python interpreter functionality.

