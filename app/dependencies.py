# app/dependencies.py

import os
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()

# Initialize Elasticsearch
es = Elasticsearch(
    hosts=[os.getenv('ES_HOST_URL')],
    http_auth=(os.getenv('ES_USERNAME'), os.getenv('ES_PASSWORD'))
)

# Initialize SentenceTransformer model
model_path = os.getenv("MODEL_PATH")
model = SentenceTransformer(model_path)

# Read NUM_RESULTS and INSTRUCTION_PROMPT from .env file
num_results = int(os.getenv("NUM_RESULTS"))
instruction_prompt = os.getenv("INSTRUCTION_PROMPT")

# Read chat model configurations
ollama_chat_endpoint = os.getenv("OLLAMA_CHAT_ENDPOINT")
ollama_chat_model = os.getenv("OLLAMA_CHAT_MODEL")
