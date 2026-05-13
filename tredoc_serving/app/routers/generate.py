from fastapi import APIRouter
from app.models import GenerateRequest, GenerateResponse
from app.util.generate import generate as generate_docstring

router = APIRouter(prefix="/model/v1", tags=["generate"])


@router.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    docstring = await generate_docstring(request.code, request.style)
    return GenerateResponse(docstring=docstring)
