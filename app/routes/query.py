# app/routes/query.py

from fastapi import APIRouter, Body
from app.models import QueryRequest
from app.dependencies import model, es, num_results, instruction_prompt, ollama_chat_endpoint, ollama_chat_model
import logging
import requests

router = APIRouter()
logger = logging.getLogger(__name__)

def search_embedding(index, query_embedding, num_results=10):
    field_name = "embedding"  # Standardize the field name
    query_body = {
        "size": num_results,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": f"cosineSimilarity(params.query_vector, '{field_name}') + 1.0",
                    "params": {"query_vector": query_embedding}
                }
            }
        }
    }

    logger.debug(f"Elasticsearch query body (excluding embeddings): {{ 'size': {num_results}, 'query': {{ 'script_score': {{ 'query': {{ 'match_all': {{ }} }}, 'script': {{ 'source': 'cosineSimilarity(params.query_vector, \\'{field_name}\\') + 1.0' }} }} }} }}")
    response = es.search(index=index, body=query_body)
    logger.debug(f"Elasticsearch response: {response}")
    return response

def generate_chat_response(prompt):
    headers = {"Content-Type": "application/json"}
    data = {
        "model": ollama_chat_model,
        "prompt": prompt,
        "temperature": 0.1,
        "stream": False
    }

    logger.info(f"Sending payload to Ollama API: {data}")  # Log the payload

    try:
        response = requests.post(ollama_chat_endpoint, headers=headers, json=data)
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

@router.post("/api/query/")
async def query(query_request: QueryRequest = Body(embed=True)):
    index_name, question, pre_msgs = query_request.index_name, query_request.question, query_request.pre_msgs

    query_embedding = model.encode(question).tolist()
    search_results = search_embedding(index_name, query_embedding, num_results)

    # Extract and log the text content from the search results
    texts_from_results = [hit['_source']['text'] for hit in search_results['hits']['hits']]
    # logger.debug(f"Texts from Elasticsearch search results: {texts_from_results}")

    # Log the number of results returned from Elasticsearch
    logger.debug(f"Number of results returned from Elasticsearch: {len(texts_from_results)}")

    matched_texts = [{"role": "system", "content": text} for text in texts_from_results]
    messages = pre_msgs + matched_texts + [{"role": "user", "content": question}]

    # Log the number of messages being sent to the LLM
    logger.debug(f"Number of messages sent to the LLM: {len(messages)}")
    # logger.debug(f"Messages sent to the LLM: {messages}")

    # Generate the prompt for the LLM and log it
    prompt = f"{instruction_prompt}\n\n" + "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    logger.debug(f"Prompt sent to the LLM: {prompt}")

    response_text = generate_chat_response(prompt)
    return {"response": response_text}
