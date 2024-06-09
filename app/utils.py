# app/utils.py

from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer
import hashlib
import regex as re
import os
from dotenv import load_dotenv
import PyPDF2
from docx import Document

# Load environment variables from .env file
load_dotenv()

# Maximum number of tokens in a chunk
max_tokens = int(os.getenv("MAX_TOKENS"))

# Load the tokenizer
tokenizer_path = os.getenv("TOKENIZER_PATH")
tokenizer = Tokenizer.from_file(os.path.join(tokenizer_path, "tokenizer.json"))

# Initialize the TextSplitter with the tokenizer and maximum tokens
splitter = TextSplitter.from_huggingface_tokenizer(tokenizer, max_tokens)

def clean_text(text):
    # Remove non-printable characters (characters not in any Unicode script)
    text = re.sub(r'[^\P{C}]+', '', text)  # \P{C} matches any character except control characters (including non-printable characters)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def split_text_semantically(text):
    # Split the document into chunks
    chunks = splitter.chunks(text)
    return chunks

def generate_unique_id(text):
    hash_object = hashlib.sha256()
    hash_object.update(text.encode('utf-8'))
    unique_id = hash_object.hexdigest()
    return unique_id

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file.file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return clean_text(text)

def extract_text_from_word(file):
    doc = Document(file.file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return clean_text(text)
