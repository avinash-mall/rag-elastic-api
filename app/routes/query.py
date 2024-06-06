# app/routes/query.py

from fastapi import APIRouter, Body
from app.models import QueryRequest
from app.dependencies import ollama_service, es
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
    response_filtered = {k: v for k, v in response.items() if k != 'embedding'}
    logger.debug(f"Elasticsearch response: {response_filtered}")
    return response

@router.post("/api/query/")
async def query(query_request: QueryRequest = Body(embed=True)):
    index_name, question, pre_msgs = query_request.index_name, query_request.question, query_request.pre_msgs

    query_embedding = ollama_service.generate_embedding(question)
    search_results = search_embedding(index_name, query_embedding)

    # Extract and log the text content from the search results
    texts_from_results = [hit['_source']['text'] for hit in search_results['hits']['hits']]
    logger.debug(f"Texts from Elasticsearch search results: {texts_from_results}")

    matched_texts = [{"role": "system", "content": text} for text in texts_from_results]
    messages = pre_msgs + matched_texts + [{"role": "user", "content": question}]

    # Log the messages being sent to Ollama API
    logger.debug(f"Messages sent to Ollama API: {messages}")

    response_text = ollama_service.generate_chat_response(messages)
    return {"response": response_text}
