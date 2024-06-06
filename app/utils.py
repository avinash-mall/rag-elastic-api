# app/utils.py

import hashlib
import spacy
from PyPDF2 import PdfReader
from docx import Document

# Initialize SpaCy
nlp = spacy.load("en_core_web_sm")

def split_text_semantically(text):
    return [sent.text.strip() for sent in nlp(text).sents]

def generate_unique_id(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def extract_text_from_pdf(file):
    reader = PdfReader(file.file)
    return ''.join([page.extract_text() for page in reader.pages])

def extract_text_from_word(file):
    doc = Document(file.file)
    return ''.join([para.text for para in doc.paragraphs])
