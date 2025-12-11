import os
import google.generativeai as genai
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import time

load_dotenv()

class GeminiAdapter:
    """
    Custom model provider adapter for Gemini API to work with the agent orchestration layer.
    """

    def __init__(self):
        """
        Initialize the Gemini adapter with API key from environment variables.
        """
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        # Configure the API key
        genai.configure(api_key=self.api_key)

        # Initialize the model
        self.model_name = "gemini-pro"  # Using gemini-pro as the default model
        self.model = genai.GenerativeModel(self.model_name)

        # Track usage for observability
        self.request_count = 0
        self.token_usage = 0

    def generate_response(self, prompt: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate a response using the Gemini API.

        Args:
            prompt: The input prompt to send to the model
            context: Additional context information (optional)

        Returns:
            Dictionary containing the response and metadata
        """
        start_time = time.time()
        self.request_count += 1

        try:
            # Prepare the generation configuration
            generation_config = {
                "temperature": 0.7,
                "max_output_tokens": 1000,
                "top_p": 0.95,
                "top_k": 40
            }

            # Generate content using the model
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )

            # Extract the text response
            text_response = response.text if response.text else "I couldn't generate a response for your query."

            # Calculate response time
            response_time = time.time() - start_time

            # Calculate approximate token usage (this is a simplified calculation)
            # In a real implementation, you'd use the actual token counts from the response
            input_tokens = len(prompt.split())
            output_tokens = len(text_response.split())
            total_tokens = input_tokens + output_tokens
            self.token_usage += total_tokens

            result = {
                "response": text_response,
                "model": self.model_name,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "response_time": response_time,
                "timestamp": time.time()
            }

            return result

        except Exception as e:
            # Calculate response time even for errors
            response_time = time.time() - start_time

            error_result = {
                "error": str(e),
                "model": self.model_name,
                "response_time": response_time,
                "timestamp": time.time()
            }

            return error_result

    def chat_generate_response(self, messages: List[Dict[str, str]], context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate a response using the Gemini API in a chat format.

        Args:
            messages: List of messages in the format {"role": "user/system/assistant", "content": "message text"}
            context: Additional context information (optional)

        Returns:
            Dictionary containing the response and metadata
        """
        start_time = time.time()
        self.request_count += 1

        try:
            # Convert messages to the format expected by Gemini
            # For now, we'll simplify by combining all messages into one prompt
            # In a real implementation, you'd use the proper chat interface
            prompt_parts = []
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                prompt_parts.append(f"{role}: {content}")

            full_prompt = "\n".join(prompt_parts)

            # Generate content using the model
            response = self.model.generate_content(full_prompt)

            # Extract the text response
            text_response = response.text if response.text else "I couldn't generate a response for your query."

            # Calculate response time
            response_time = time.time() - start_time

            # Calculate approximate token usage
            input_tokens = len(full_prompt.split())
            output_tokens = len(text_response.split())
            total_tokens = input_tokens + output_tokens
            self.token_usage += total_tokens

            result = {
                "response": text_response,
                "model": self.model_name,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "response_time": response_time,
                "timestamp": time.time()
            }

            return result

        except Exception as e:
            # Calculate response time even for errors
            response_time = time.time() - start_time

            error_result = {
                "error": str(e),
                "model": self.model_name,
                "response_time": response_time,
                "timestamp": time.time()
            }

            return error_result

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the configured model.

        Returns:
            Dictionary containing model information
        """
        try:
            model_info = genai.models.get(model_name=self.model_name)
            return {
                "name": model_info.name,
                "version": model_info.version,
                "display_name": model_info.display_name,
                "description": model_info.description,
                "input_token_limit": model_info.input_token_limit,
                "output_token_limit": model_info.output_token_limit,
                "supported_generation_methods": model_info.supported_generation_methods
            }
        except Exception as e:
            return {
                "error": str(e),
                "name": self.model_name
            }

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics for observability.

        Returns:
            Dictionary containing usage statistics
        """
        return {
            "request_count": self.request_count,
            "total_tokens_used": self.token_usage,
            "model": self.model_name
        }

    def transform_request(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform request from OpenAI format to Gemini format.

        Args:
            input_data: Input data in OpenAI format

        Returns:
            Transformed input data in Gemini format
        """
        # This is a simplified transformation
        # In a real implementation, this would handle more complex mappings
        transformed = {
            "prompt": input_data.get("prompt", ""),
            "messages": input_data.get("messages", []),
            "temperature": input_data.get("temperature", 0.7),
            "max_tokens": input_data.get("max_tokens", 1000),
            "top_p": input_data.get("top_p", 0.95),
            "top_k": input_data.get("top_k", 40)
        }
        return transformed

    def transform_response(self, gemini_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform response from Gemini format to OpenAI format.

        Args:
            gemini_response: Response from Gemini API

        Returns:
            Transformed response in OpenAI format
        """
        # This is a simplified transformation
        # In a real implementation, this would handle more complex mappings
        transformed = {
            "choices": [
                {
                    "text": gemini_response.get("response", ""),
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": gemini_response.get("input_tokens", 0),
                "completion_tokens": gemini_response.get("output_tokens", 0),
                "total_tokens": gemini_response.get("total_tokens", 0)
            },
            "model": gemini_response.get("model", self.model_name),
            "object": "text_completion"
        }
        return transformed


# Example usage
if __name__ == "__main__":
    try:
        adapter = GeminiAdapter()
        print("Gemini Adapter initialized successfully")

        # Test basic generation
        result = adapter.generate_response("What is machine learning?")
        print(f"Response: {result['response'][:100]}...")
        print(f"Tokens used: {result['total_tokens']}")
        print(f"Response time: {result['response_time']:.2f}s")

        # Show usage stats
        stats = adapter.get_usage_stats()
        print(f"Usage stats: {stats}")

    except ValueError as e:
        print(f"Error initializing Gemini Adapter: {e}")
        print("Make sure GEMINI_API_KEY is set in environment variables")