import os
import openai
import google.generativeai as genai
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class LLMClient:
    """
    Production-ready LLM client supporting both OpenAI and Gemini APIs.
    """
    
    def __init__(self):
        self.openai_client = None
        self.gemini_client = None
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize API clients based on available environment variables."""
        # Initialize OpenAI client
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
        # Initialize Gemini client
        gemini_api_key = os.getenv("GOOGLE_API_KEY")
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.gemini_client = genai.GenerativeModel('gemini-1.5-flash')
    
    def call_llm(self, prompt: str, model: str = "gemini-1.5-flash", **kwargs) -> str:
        """
        Call the specified LLM model with proper error handling.
        
        Args:
            prompt (str): The prompt to send to the LLM
            model (str): Model identifier (e.g., "gemini-1.5-flash", "gpt-4")
            **kwargs: Additional parameters for the LLM call
            
        Returns:
            str: The LLM response content
            
        Raises:
            RuntimeError: If no available LLM client or API call fails
        """
        if not self.openai_client and not self.gemini_client:
            raise RuntimeError("No LLM API keys configured. Please set OPENAI_API_KEY or GOOGLE_API_KEY")
        
        try:
            if "gemini" in model.lower() and self.gemini_client:
                return self._call_gemini(prompt, model, **kwargs)
            elif self.openai_client:
                return self._call_openai(prompt, model, **kwargs)
            else:
                # Fallback to available client
                if self.gemini_client:
                    return self._call_gemini(prompt, "gemini-1.5-flash", **kwargs)
                else:
                    return self._call_openai(prompt, "gpt-4", **kwargs)
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {str(e)}")
    
    def _call_gemini(self, prompt: str, model: str, **kwargs) -> str:
        """Call Gemini API."""
        response = self.gemini_client.generate_content(prompt)
        return response.text
    
    def _call_openai(self, prompt: str, model: str, **kwargs) -> str:
        """Call OpenAI API."""
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 2000),
            **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]}
        )
        return response.choices[0].message.content

# Global instance
llm_client = LLMClient()

def call_llm(prompt: str, model: str = "gemini-1.5-flash", **kwargs) -> str:
    """Convenience function for backward compatibility."""
    return llm_client.call_llm(prompt, model, **kwargs)

if __name__ == '__main__':
    # Test the client
    try:
        test_prompt = "Hello, world! This is a test."
        print(f"Sending test prompt: '{test_prompt}'")
        response = call_llm(test_prompt)
        print(f"Received response: {response}")
    except Exception as e:
        print(f"Error: {e}") 