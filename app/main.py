# app/main.py

import uvicorn
import logging
from app import app
from app.routes import index, query, manage_index, upload

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Include routers
app.include_router(index.router)
app.include_router(query.router)
app.include_router(manage_index.router)
app.include_router(upload.router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
