"""Request models for the MioBot web admin API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class TokenLoginRequest(BaseModel):
    token: str = Field(min_length=1, max_length=256)


class MemorySummaryUpdateRequest(BaseModel):
    memory_text: str = Field(default="", max_length=8000)


class UserFactUpdateRequest(BaseModel):
    fact_type: Optional[str] = Field(default=None, max_length=80)
    fact_text: Optional[str] = Field(default=None, max_length=4000)
    confidence: Optional[float] = None


class GlobalFactCreateRequest(BaseModel):
    fact_type: str = Field(default="note", max_length=80)
    fact_text: str = Field(min_length=1, max_length=4000)
    confidence: float = 0.9


class GlobalFactUpdateRequest(BaseModel):
    fact_type: Optional[str] = Field(default=None, max_length=80)
    fact_text: Optional[str] = Field(default=None, max_length=4000)
    confidence: Optional[float] = None
