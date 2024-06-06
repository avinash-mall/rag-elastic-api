# app/models.py

from pydantic import BaseModel, Field
from typing import List, Optional

class IndexRequest(BaseModel):
    index_name: str = Field(..., description="Name of the Elasticsearch index")
    text: str = Field(..., description="Text to be indexed")

class QueryRequest(BaseModel):
    index_name: str = Field(..., description="Name of the Elasticsearch index")
    question: str = Field(..., description="Query question")
    pre_msgs: Optional[List[dict]] = Field(default=[], description="Previous messages")

class CreateIndexRequest(BaseModel):
    index_name: str = Field(..., description="Name of the Elasticsearch index")
    dims: int = Field(..., description="Dimensions for the embeddings")
# app/models.py

from pydantic import BaseModel, Field
from typing import List, Optional

class IndexRequest(BaseModel):
    index_name: str = Field(..., description="Name of the Elasticsearch index")
    text: str = Field(..., description="Text to be indexed")

class QueryRequest(BaseModel):
    index_name: str = Field(..., description="Name of the Elasticsearch index")
    question: str = Field(..., description="Query question")
    pre_msgs: Optional[List[dict]] = Field(default=[], description="Previous messages")

class CreateIndexRequest(BaseModel):
    index_name: str = Field(..., description="Name of the Elasticsearch index")
    dims: int = Field(..., description="Dimensions for the embeddings")
