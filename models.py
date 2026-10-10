from pydantic import BaseModel, Field
from typing import Optional

class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    k: int = Field(20, ge=1, le=100)
    explain: bool = False
    rewrite: bool = False

class RecommendationItem(BaseModel):
    item_id: str
    title: str
    score: float
    reason: list[str] = []
    explain: Optional[dict] = None
