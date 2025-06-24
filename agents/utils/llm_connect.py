from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient
import logging
import requests
import time

# Configuration variable to decide which LLM to use
USE_LOCAL_LLM = True  # Set to False to use OpenAI API

# Setup basic logging
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
'''
import os
import sys

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_dir)
'''



# Local LLM implementation (Ollama)
class OllamaClient:
    def __init__(self, model_name="tinyllama", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.model_info = {"name": model_name}
        self.function_calling = False  # Attribute

    def __getitem__(self, key):
        # Cover dict-like access
        if key == "function_calling":
            return self.function_calling
        elif key == "name":
            return self.model_name
        elif key == "model_info":
            return self.model_info
        raise KeyError(f"Unsupported key requested: {key}")

    def create(self, messages, **kwargs):
        prompt = self._convert_messages_to_prompt(messages)
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model_name, "prompt": prompt}
            )
            if response.status_code == 200:
                return {"choices": [{"message": {"content": response.json().get("response", "")}}]}
            else:
                raise Exception(f"Ollama error: {response.status_code} - {response.text}")
        except Exception as e:
            logging.error(f"Ollama error: {str(e)}")
            return {"choices": [{"message": {"content": "Error connecting to local LLM"}}]}

    def _convert_messages_to_prompt(self, messages):
        prompt = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            prompt += f"{role}: {content}\n"
        return prompt


class WrappedOllamaClient:
    def __init__(self, client: OllamaClient):
        self._client = client
        self.model_info = {"name": client.model_name}
        self.function_calling = False

    def __getitem__(self, key):
        if key == "function_calling":
            return self.function_calling
        elif key == "name":
            return self._client.model_name
        elif key == "model_info":
            return self.model_info
        raise KeyError(key)

    def create(self, messages, **kwargs):
        return self._client.create(messages, **kwargs)


# OpenAI client with retry logic
def get_openai_client(max_retries=3, retry_delay=2):
    for attempt in range(max_retries):
        try:
            return OpenAIChatCompletionClient(
                model="gpt-4o-2024-11-20",
                timeout=60  # Increase timeout
            )
        except Exception as e:
            logging.error(f"Attempt {attempt+1}: Error initializing model client: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise
