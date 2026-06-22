from pydantic import BaseModel
from typing import Optional, Literal
from datetime import datetime


class ProviderConfig(BaseModel):
    type: Literal["openai_compatible", "anthropic", "gemini"]
    api_key: str
    model: str
    base_url: Optional[str] = None  # only used for openai_compatible


class ChatRequest(BaseModel):
    session_id: str
    message: str
    provider: ProviderConfig


class SessionCreate(BaseModel):
    title: Optional[str] = "New Chat"


class SessionResponse(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class MessageResponse(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}
