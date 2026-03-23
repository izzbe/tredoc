import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://tredoc:tredoc@localhost:5432/tredoc"
)
engine = create_async_engine(DATABASE_URL)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

async def get_db():
    async with SessionLocal() as session:
        yield session

