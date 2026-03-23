from fastapi import FastAPI
from app.routers import generate, health
app = FastAPI()

app.include_router(generate.router)
app.include_router(health.router)