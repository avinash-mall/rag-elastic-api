# app/dependencies.py

import os
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from app.services import OllamaService

# Load environment variables from .env file
load_dotenv()

# Initialize Elasticsearch
es = Elasticsearch(
    hosts=[os.getenv('ES_HOST_URL')],
    http_auth=(os.getenv('ES_USERNAME'), os.getenv('ES_PASSWORD'))
)

# Initialize OllamaService
ollama_service = OllamaService(
    embed_endpoint=os.getenv("OLLAMA_EMBEDDING_ENDPOINT"),
    chat_endpoint=os.getenv("OLLAMA_CHAT_ENDPOINT"),
    embed_model=os.getenv("OLLAMA_EMBED_MODEL"),
    chat_model=os.getenv("OLLAMA_CHAT_MODEL")
)

# Read NUM_RESULTS and INSTRUCTION_PROMPT from .env file
num_results = int(os.getenv("NUM_RESULTS", 10))
instruction_prompt = os.getenv("INSTRUCTION_PROMPT", "Provide a well-reasoned and informative response to the query.")
