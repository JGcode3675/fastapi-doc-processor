from datetime import datetime
from pydantic import BaseModel, ConfigDict


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
