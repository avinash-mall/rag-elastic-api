# app/routes/manage_index.py

from fastapi import APIRouter, HTTPException, Body
from app.dependencies import es
from app.models import CreateIndexRequest

router = APIRouter()

@router.post("/api/create_index/")
async def create_index(request: CreateIndexRequest = Body(embed=True)):
    index_name = request.index_name
    dims = request.dims
    field_name = "embedding"  # Standardize the field name

    if es.indices.exists(index=index_name):
        raise HTTPException(status_code=400, detail="Index already exists")

    # Create the index with specific dimensions
    mappings = {
        "properties": {
            field_name: {
                "type": "dense_vector",
                "dims": dims,
                "index": "true",
                "similarity": "cosine",
            }
        }
    }

    es.indices.create(index=index_name, body={"mappings": mappings})
    return {"message": f"Index '{index_name}' with {dims} dimensions created successfully"}

@router.delete("/api/delete_index/")
async def delete_index(index_name: str = Body(embed=True)):
    if not es.indices.exists(index=index_name):
        raise HTTPException(status_code=404, detail="Index not found")
    es.indices.delete(index=index_name)
    return {"message": f"Index '{index_name}' deleted successfully"}

@router.get("/api/list_indexes/")
async def list_indexes():
    indexes = es.indices.get_alias(index="*")
    filtered_indexes = [index for index in indexes.keys() if not index.startswith('.')]
    return {"indexes": filtered_indexes}
