# this is the main model that the user will communicate with
# all conversations will be had using this model
# will be using chatgpt 3.5 for now...

from openai import OpenAI
from typing import List, Dict

class LLM:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo-0125"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 150) -> str:
        try:
            response = self.client.chat.completions.create(model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens)
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in generating response: {e}")
            return ""

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 150) -> str:
        try:
            response = self.client.chat.completions.create(model=self.model,
            messages=messages,
            max_tokens=max_tokens)
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in chat: {e}")
            return ""