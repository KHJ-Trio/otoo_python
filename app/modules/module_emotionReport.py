from fastapi import HTTPException
from pydantic import BaseModel

class messagesRequest(BaseModel):
    messages: str

async def generate_messages_response(client, messages_request: messagesRequest) -> str:
    return await client.generate_messages_response(messages_request)

