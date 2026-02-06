from datetime import datetime
from pydantic import BaseModel, ConfigDict


# --- Document Schemas ---

class DocumentBase(BaseModel):
    filename: str
    content_type: str | None = None
    file_size: int | None = None


class DocumentCreate(DocumentBase):
    pass


class DocumentResponse(DocumentBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    status: str
    summary: str | None = None
    created_at: datetime
    updated_at: datetime


class UploadResponse(BaseModel):
    message: str
    document: DocumentResponse


class SearchResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    summary: str | None = None
    similarity: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    count: int


# --- Stats ---

class StatsResponse(BaseModel):
    documents: int
    memories: int
    status: str


# --- Chat Schemas ---

class ChatRequest(BaseModel):
    query: str
    model: str = "moltbot"
    system_prompt: str | None = None
    session_id: int | None = None


class ChatResponse(BaseModel):
    response: str
    model: str
    tokens_used: int | None = None
    session_id: int | None = None


class ChatMessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    tool_calls: dict | None = None
    tokens_used: int | None = None
    created_at: datetime


class ChatSessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str | None = None
    system_prompt: str | None = None
    created_at: datetime


# --- Briefing Schemas ---

class BriefingRequest(BaseModel):
    legs: list[str] = ["market", "news", "osint", "tech"]


class BriefingResponse(BaseModel):
    markdown_report: str
    timestamp: str
    document_count: int
