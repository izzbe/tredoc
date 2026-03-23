from fastapi import APIRouter

router = APIRouter(prefix="/health")

@router.get("")
async def generate():
    return {"response": "not implemented"}

