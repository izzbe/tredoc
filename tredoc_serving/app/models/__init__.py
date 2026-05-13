from pydantic import BaseModel


class GenerateRequest(BaseModel):
    code: str
    style: str = "google"


class GenerateResponse(BaseModel):
    docstring: str
