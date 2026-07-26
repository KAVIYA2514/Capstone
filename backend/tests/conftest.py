import pytest_asyncio
from app.core.db import AsyncSessionLocal


@pytest_asyncio.fixture
async def db_session():
    async with AsyncSessionLocal() as session:
        yield session
