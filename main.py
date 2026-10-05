from contextlib import asynccontextmanager
from fastapi import FastAPI
from models import SwipeEvent, SearchRequest, SurveyRequest, RecommendationItem
import llm
import retrieval

@asynccontextmanager
async def lifespan(app):
    yield
    llm.close()

app = FastAPI(lifespan=lifespan)

@app.post("/survey/start")
def survey_start(req: SurveyRequest):
    return retrieval.start_survey(req.user_id, req.survey_size)

@app.post("/swipe")
def swipe(event: SwipeEvent):
    return retrieval.swipe(event)

@app.post("/search")
def search(req: SearchRequest):
    result = retrieval.search(req.user_id, req.query, req.k, req.explain, req.rewrite)
    items = [RecommendationItem(item_id=item['item_id'], title=item['title'], score=item['score'],
                                preference_boost=item['preference_boost'], reason=item['reason'],
                                explain=item.get('explain')) for item in result['items']]
    return {
        'user_id': req.user_id,
        'items': items,
        'model_version': 'preference-v1',
        'cached': False,
        'preference_confidence': result['preference_confidence'],
        'learned_weights': result['learned_weights'],
        'rewritten_query': result['rewritten_query'],
        'llm_used': result['llm_used'],
        'llm_cached': result['llm_cached']
    }

@app.get("/user/{user_id}/profile")
def user_profile(user_id: str):
    return retrieval.get_user_profile(user_id)

if __name__ == "__main__":
    print('testing')
