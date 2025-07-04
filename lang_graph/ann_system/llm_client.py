import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the OpenAI client with the API key from environment variables
# It's important to have OPENAI_API_KEY set in your .env file or environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

client = openai.OpenAI(api_key=api_key)

def call_llm(prompt: str, model: str = "gemini-4o") -> str:
    """
    A utility function to call the specified LLM model.
    
    Args:
        prompt (str): The prompt to send to the LLM.
        model (str): The model to use (e.g., "gemini-4o", "gemini-3.5-turbo").
        
    Returns:
        str: The content of the LLM's response.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000,
        )
        content = response.choices[0].message.content
        return content if content else "Error: Empty response from LLM."
    except Exception as e:
        print(f"An error occurred while calling the LLM: {e}")
        return f"Error: Could not get a response from the LLM. Details: {e}"

if __name__ == '__main__':
    # Example usage:
    # Make sure you have a .env file with your OPENAI_API_KEY or have it set in your environment
    test_prompt = "Hello, world! This is a test."
    print(f"Sending test prompt: '{test_prompt}'")
    response_content = call_llm(test_prompt)
    print(f"Received response: {response_content}") 