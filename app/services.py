# app/services.py

import requests
import logging

logger = logging.getLogger(__name__)

class OllamaService:
    def __init__(self, embed_endpoint, chat_endpoint, embed_model, chat_model):
        self.embed_endpoint = embed_endpoint
        self.chat_endpoint = chat_endpoint
        self.embed_model = embed_model
        self.chat_model = chat_model

        if not all([self.embed_endpoint, self.chat_endpoint, self.embed_model, self.chat_model]):
            raise ValueError("Ollama API configurations are not set properly")

    def generate_embedding(self, text):
        response = requests.post(self.embed_endpoint, headers={"Content-Type": "application/json"}, json={"model": self.embed_model, "prompt": text})
        response.raise_for_status()
        return self._parse_response(response, 'embedding')

    def generate_chat_response(self, messages):
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        response = requests.post(self.chat_endpoint, headers={"Content-Type": "application/json"}, json={"model": self.chat_model, "prompt": prompt, "temperature": 0.7, "stream": False})
        response.raise_for_status()
        return self._parse_response(response, 'response')

    @staticmethod
    def _parse_response(response, key):
        try:
            response_json = response.json()
            return response_json[key]
        except (KeyError, ValueError):
            raise ValueError(f"Error parsing {key} from response: {response.text}")
