import os
import hashlib
import requests
import logging
from dotenv import load_dotenv
from typing import List, Optional, Annotated
from fastapi_offline import FastAPIOffline
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from elasticsearch import Elasticsearch

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Utility functions for text processing
import spacy

nlp = spacy.load("en_core_web_sm")

def split_text_semantically(text):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences

def generate_unique_id(text):
    hash_object = hashlib.sha256()
    hash_object.update(text.encode('utf-8'))
    unique_id = hash_object.hexdigest()
    return unique_id

# Service for interacting with Ollama API
class OllamaService:
    def __init__(self):
        self.ollama_embedding_endpoint = os.getenv("OLLAMA_EMBEDDING_ENDPOINT")
        if not self.ollama_embedding_endpoint:
            raise ValueError("OLLAMA_EMBEDDING_ENDPOINT is not set")
        self.ollama_chat_endpoint = os.getenv("OLLAMA_CHAT_ENDPOINT")
        if not self.ollama_chat_endpoint:
            raise ValueError("OLLAMA_CHAT_ENDPOINT is not set")
        self.embed_model_name = os.getenv("OLLAMA_EMBED_MODEL")
        if not self.embed_model_name:
            raise ValueError("OLLAMA_EMBED_MODEL is not set")
        self.chat_model_name = os.getenv("OLLAMA_CHAT_MODEL")
        if not self.chat_model_name:
            raise ValueError("OLLAMA_CHAT_MODEL is not set")

    def generate_embedding(self, text):
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.embed_model_name,
            "prompt": text
        }
        try:
            response = requests.post(self.ollama_embedding_endpoint, headers=headers, json=data)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error from Ollama API: {e}")

        try:
            response_json = response.json()
            embedding = response_json['embedding']
            if not isinstance(embedding, list) or not all(isinstance(i, float) for i in embedding):
                raise ValueError("Invalid embedding format received from Ollama API")
            return embedding
        except (KeyError, ValueError) as e:
            raise ValueError(f"Error parsing embedding from response: {e}\nResponse text: {response.text}")

    def generate_chat_response(self, messages):
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.chat_model_name,
            "prompt": prompt,
            "temperature": 0.7,
            "stream": False
        }
        
        logger.info(f"Sending payload to Ollama API: {data}")  # Log the payload

        try:
            response = requests.post(self.ollama_chat_endpoint, headers=headers, json=data)
            response.raise_for_status()
            logger.info(f"Ollama API response status: {response.status_code}")  # Log the status code
        except requests.exceptions.RequestException as e:
            logger.error(f"Error from Ollama API: {e}")
            raise ValueError(f"Error from Ollama API: {e}")

        try:
            response_json = response.json()
            logger.info(f"Full response JSON from Ollama API: {response_json}")  # Log the full response JSON
            response_text = response_json.get('response', '').strip()
            logger.info(f"Ollama API response text: {response_text}")  # Log the response text
            return response_text
        except (KeyError, ValueError) as e:
            logger.error(f"Error parsing response from Ollama API: {e}\nResponse text: {response.text}")
            raise ValueError(f"Error parsing response from Ollama API: {e}\nResponse text: {response.text}")

ollama_service = OllamaService()

# Set ES properties
es_host_url = os.getenv('ES_HOST_URL')
es_username = os.getenv('ES_USERNAME')
es_password = os.getenv('ES_PASSWORD')

# Initialize Elasticsearch with authentication
es = Elasticsearch(
    hosts=[es_host_url],
    http_auth=(es_username, es_password)
)

def index_document(index, id, body, hard_refresh=False):
    if hard_refresh:
        # Index the document
        es.index(index=index, id=id, body=body)
        logger.info(f"hard indexed - {id}")
    else:
        indexed = already_indexed(id, index)
        if not indexed:
            # Index the document
            es.index(index=index, id=id, body=body)
            logger.info(f"indexed - {id}")
        else:
            logger.info(f"already indexed - {id}")

def already_indexed(id, index):
    # Define Elasticsearch script score query
    body = {
        "size": 1,
        "query": {
            "match": {
                "_id": id
            }
        }
    }

    # Execute the query
    res = es.search(index=index, body=body)
    if res['hits']['total']['value'] > 0:
        return True
    return False

def search_embedding(index, query_embedding, num_results=10):
    try:
        # Define Elasticsearch script score query
        body = {
            "size": num_results,
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {
                            "query_vector": query_embedding
                        }
                    }
                }
            }
        }

        # Execute the query
        res = es.search(index=index, body=body)
        logger.info(f"Search results: {res}")  # Log the search results
        return res
    except Exception as e:
        logger.error(f"Error executing search: {e}")
        return None

# FastAPI app setup
app = FastAPIOffline()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request bodies
class IndexRequest(BaseModel):
    index_name: str = Field(..., description="Name of the Elasticsearch index")
    text: str = Field(..., description="Text to be indexed")

class QueryRequest(BaseModel):
    index_name: str = Field(..., description="Name of the Elasticsearch index")
    question: str = Field(..., description="Query question")
    pre_msgs: Optional[List[dict]] = Field(default=[], description="Previous messages")

@app.get("/")
async def hello():
    return {"message": "Hello from RAG Application"}

@app.post("/api/index/")
async def index_text(index_request: Annotated[IndexRequest, Body(embed=True)]):
    index = index_request.index_name
    text = index_request.text

    chunks = split_text_semantically(text)

    for chunk in chunks:
        if chunk and len(chunk) > 0:
            embedding = ollama_service.generate_embedding(chunk)
            doc_id = generate_unique_id(chunk)
            body = {
                "text": chunk,
                "embedding": embedding
            }
            index_document(index, doc_id, body)

    return {"message": "Text indexed successfully"}

@app.post("/api/query/")
async def query(query_request: Annotated[QueryRequest, Body(embed=True)]):
    index_name = query_request.index_name
    question = query_request.question
    pre_msgs = query_request.pre_msgs

    query_embedding = ollama_service.generate_embedding(question)
    search_results = search_embedding(index_name, query_embedding)

    # Collect matched texts and format them as messages
    matched_texts = []
    for hit in search_results['hits']['hits']:
        matched_texts.append({"role": "system", "content": hit['_source']['text']})

    logger.info(f"Matched texts: {matched_texts}")  # Log matched texts

    # Combine previous messages, matched texts, and the current question
    messages = pre_msgs + matched_texts + [{"role": "user", "content": question}]
    logger.info(f"Messages for Ollama: {messages}")  # Log messages sent to Ollama API

    # Generate the response using Ollama service
    response_text = ollama_service.generate_chat_response(messages)
    logger.info(f"Final response text: {response_text}")  # Log the final response

    return {"response": response_text}

# Elasticsearch index management endpoints
@app.post("/api/create_index/")
async def create_index(index_name: Annotated[str, Body(embed=True)]):
    try:
        if es.indices.exists(index=index_name):
            raise HTTPException(status_code=400, detail="Index already exists")
        es.indices.create(index=index_name)
        return {"message": f"Index '{index_name}' created successfully"}
    except Exception as e:
        logger.error(f"Error creating index: {e}")
        raise HTTPException(status_code=500, detail="Error creating index")

@app.delete("/api/delete_index/")
async def delete_index(index_name: Annotated[str, Body(embed=True)]):
    try:
        if not es.indices.exists(index=index_name):
            raise HTTPException(status_code=404, detail="Index not found")
        es.indices.delete(index=index_name)
        return {"message": f"Index '{index_name}' deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting index: {e}")
        raise HTTPException(status_code=500, detail="Error deleting index")

@app.get("/api/list_indexes/")
async def list_indexes():
    try:
        indexes = es.indices.get_alias(index="*")
        filtered_indexes = [index for index in indexes.keys() if not index.startswith('.')]
        return {"indexes": filtered_indexes}
    except Exception as e:
        logger.error(f"Error listing indexes: {e}")
        raise HTTPException(status_code=500, detail="Error listing indexes")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
