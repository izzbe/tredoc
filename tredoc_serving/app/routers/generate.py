from fastapi import APIRouter

router = APIRouter(prefix="/model/v1", tags=["generate"])

@router.get("/generate")
async def generate():
    return {"response": "not implemented"}

