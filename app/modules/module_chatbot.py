from pydantic import BaseModel
from typing import List


class modeRequest(BaseModel):
    mode: str

class RecentMessagesRequest(BaseModel):
    RecentMessages: List[str]

class FullRequest(BaseModel):
    mode_request: modeRequest
    recent_messages_request: RecentMessagesRequest



async def generate_chat_response(client, mode_request, recent_messages_request):
    return await client.generate_chat_response(mode_request, recent_messages_request)