# Semantic Text Processing with FastAPI, Ollama API, and Elasticsearch

This repository contains a FastAPI application for semantic text processing using SpaCy, Ollama API, and Elasticsearch. The application provides endpoints for indexing text, querying indexed text, and managing Elasticsearch indexes.

## Features

- Semantic text splitting using SpaCy
- Text embedding generation using Ollama API
- Chat response generation using Ollama API
- Text indexing and searching with Elasticsearch
- API endpoints for managing indexes and querying data

## Setup

### Prerequisites

- Python 3.7+
- Elasticsearch instance (You can use deviantony's repo for easy setup)
   ```bash
   git clone https://github.com/deviantony/docker-elk.git
   cd docker-elk
   docker compose up setup
   docker compose up -d
   ```
- Ollama running with encoding model and LLM

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/avinash-mall/rag-elastic-api.git
   cd rag-elastic-api
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**

   Create a `.env` file in the root directory of the project and add the following variables:

   ```env
   OLLAMA_EMBEDDING_ENDPOINT=<your_ollama_embedding_endpoint>
   OLLAMA_CHAT_ENDPOINT=<your_ollama_chat_endpoint>
   OLLAMA_EMBED_MODEL=<your_ollama_embed_model>
   OLLAMA_CHAT_MODEL=<your_ollama_chat_model>
   ES_HOST_URL=<your_elasticsearch_host_url>
   ES_USERNAME=<your_elasticsearch_username>
   ES_PASSWORD=<your_elasticsearch_password>
   ```
Replace the placeholder values such as `<your_ollama_embedding_endpoint>`, `<your_elasticsearch_host_url>`, and others with the actual values you use in your environment.

### Running the Application

1. **Start the FastAPI server:**

   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8081
   ```

2. **Access the API documentation:**

   Open your browser and navigate to `http://localhost:8081/docs` to view the interactive API documentation.

## API Endpoints

### General

- `GET /`: Returns a simple greeting message.

### Indexing

- `POST /api/index/`: Indexes provided text by splitting it semantically and generating embeddings.

  **Request Body:**
  ```json
  {
    "index_name": "your_index_name",
    "text": "Text to be indexed"
  }
  ```

### Querying

- `POST /api/query/`: Queries indexed texts based on a question and generates a response.

  **Request Body:**
  ```json
  {
    "index_name": "your_index_name",
    "question": "Your query question",
    "pre_msgs": [{"role": "user", "content": "Previous message"}]
  }
  ```

### Index Management

- `POST /api/create_index/`: Creates a new Elasticsearch index.

  **Request Body:**
  ```json
  {
    "index_name": "your_index_name"
  }
  ```

- `DELETE /api/delete_index/`: Deletes an existing Elasticsearch index.

  **Request Body:**
  ```json
  {
    "index_name": "your_index_name"
  }
  ```

- `GET /api/list_indexes/`: Lists all Elasticsearch indexes.

## Contributing

1. **Fork the repository.**
2. **Create a new branch:**

   ```bash
   git checkout -b my-new-feature
   ```

3. **Commit your changes:**

   ```bash
   git commit -am 'Add some feature'
   ```

4. **Push to the branch:**

   ```bash
   git push origin my-new-feature
   ```

5. **Submit a pull request.**

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
