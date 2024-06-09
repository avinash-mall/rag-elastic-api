# download_tokenizer.py

from transformers import AutoTokenizer
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

tokenizer_name = os.getenv("TOKENIZER_NAME")
tokenizer_path = os.getenv("TOKENIZER_PATH")

# Download and save the tokenizer locally
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
os.makedirs(tokenizer_path, exist_ok=True)
tokenizer.save_pretrained(tokenizer_path)
