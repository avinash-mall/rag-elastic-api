# app/routes/query.py

from fastapi import APIRouter, Body
from app.models import QueryRequest
from app.dependencies import ollama_service, es, num_results, instruction_prompt
import logging

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

@router.post("/api/query/")
async def query(query_request: QueryRequest = Body(embed=True)):
    index_name, question, pre_msgs = query_request.index_name, query_request.question, query_request.pre_msgs

    query_embedding = ollama_service.generate_embedding(question)
    search_results = search_embedding(index_name, query_embedding, num_results)

    # Extract and log the text content from the search results
    texts_from_results = [hit['_source']['text'] for hit in search_results['hits']['hits']]
    logger.debug(f"Texts from Elasticsearch search results: {texts_from_results}")

    # Log the number of results returned from Elasticsearch
    logger.debug(f"Number of results returned from Elasticsearch: {len(texts_from_results)}")

    matched_texts = [{"role": "system", "content": text} for text in texts_from_results]
    messages = pre_msgs + matched_texts + [{"role": "user", "content": question}]

    # Log the number of messages being sent to Ollama API
    logger.debug(f"Number of messages sent to Ollama API: {len(messages)}")
    logger.debug(f"Messages sent to Ollama API: {messages}")

    # Generate the prompt for the LLM and log it
    prompt = f"{instruction_prompt}\n\n" + "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    logger.debug(f"Prompt sent to Ollama API: {prompt}")

    response_text = ollama_service.generate_chat_response(messages)
    return {"response": response_text}
