import os
import json
import httpx
import argparse
import asyncio
from pathlib import Path

# This is a self-contained script with no dependencies on the mcp_agent framework.
# It uses httpx to directly call the OpenAI API.

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

def create_prompt(description: str, server_name: str) -> str:
    # Re-adding the explicit JSON instruction at the end of the prompt
    # to satisfy the API requirement for 'json_object' response format.
    return f"""
    You are an expert software engineer specializing in Node.js and Express.
    Your task is to generate the boilerplate code for a new MCP (Model Context Protocol) server based on a high-level description.

    **MCP Server Description:** "{description}"
    **Server Name:** "{server_name}"

    You MUST generate two files: `package.json` and `index.js`.
    
    ... (rest of the prompt rules are the same) ...

    **CRITICAL: Your entire output MUST be a single, valid JSON object.**
    This JSON object must have exactly two keys: "package_json" and "index_js".
    The value for each key must be the complete, formatted string content of the corresponding file.
    Do not include any other text, explanations, or markdown formatting outside of the main JSON object.
    Ensure the final output is a raw JSON string.
    """

async def generate_code(description: str, server_name: str) -> dict:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    
    prompt = create_prompt(description, server_name)
    
    payload = {
        "model": "gpt-5-mini-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(OPENAI_API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"OpenAI API request failed with status {response.status_code}: {response.text}")

    response_data = response.json()
    message_content = response_data["choices"][0]["message"]["content"]
    
    try:
        return json.loads(message_content)
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON from LLM response: {message_content}")
        raise e

async def main():
    parser = argparse.ArgumentParser(description="Self-Contained Code Generator for MCP Servers.")
    parser.add_argument("--description", required=True, help="Description of the MCP server.")
    parser.add_argument("--path", required=True, help="Target directory for the new server.")
    args = parser.parse_args()

    target_path = Path(args.path)
    server_name = target_path.name

    try:
        print(f"🤖 Generating code for '{server_name}'...")
        generated_code = await generate_code(args.description, server_name)

        # File saving logic
        package_json = generated_code.get("package_json")
        index_js = generated_code.get("index_js")

        if not (package_json and index_js):
            print("❌ LLM response was missing required code content.")
            return

        target_path.mkdir(parents=True, exist_ok=True)
        (target_path / "package.json").write_text(package_json, encoding='utf-8')
        (target_path / "index.js").write_text(index_js, encoding='utf-8')

        print(f"✅ Successfully generated files in: {args.path}")
        print("\n🎉 Workflow finished successfully!")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 