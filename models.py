from pydantic import BaseModel
from typing import Literal, Optional

class SwipeEvent(BaseModel):
    user_id: str
    item_id: str
    direction: Literal['like', 'dislike']
    session_id: str
    timestamp: Optional[int] = None

class SearchRequest(BaseModel):
    user_id: str
    query: str
    k: int = 20
    explain: bool = False

class SurveyRequest(BaseModel):
    user_id: str
    survey_size: int = 20

class RecommendationItem(BaseModel):
    item_id: str
    title: str
    score: float
    preference_boost: float
    reason: list[str] = []

class UserPreferences(BaseModel):
    user_id: str
    swipe_count: int
    preference_confidence: float
    preference_vector: Optional[list[float]] = None
    genre_preferences: dict[str, float] = {}
    tag_preferences: dict[str, float] = {}
    learned_weights: dict[str, float] = {}
