# app/__init__.py

from fastapi_offline import FastAPIOffline
from fastapi.middleware.cors import CORSMiddleware

app = FastAPIOffline()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
