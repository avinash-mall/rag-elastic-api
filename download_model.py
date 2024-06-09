# download_model.py

from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

model_name = os.getenv("MODEL_NAME")
model_path = os.getenv("MODEL_PATH")

# Download and save the model locally
model = SentenceTransformer(model_name)
os.makedirs(model_path, exist_ok=True)
model.save(model_path)
