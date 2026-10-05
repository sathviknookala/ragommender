from fastapi import FastAPI
from models import SwipeEvent, SearchRequest, SurveyRequest, RecommendationItem
import retrieval

app = FastAPI()

@app.post("/survey/start")
def survey_start(req: SurveyRequest):
    return retrieval.start_survey(req.user_id, req.survey_size)

@app.post("/swipe")
def swipe(event: SwipeEvent):
    return retrieval.swipe(event)

@app.post("/search")
def search(req: SearchRequest):
    result = retrieval.search(req.user_id, req.query, req.k)
    items = [RecommendationItem(item_id=item['item_id'], title=item['title'], score=item['score'],
                                preference_boost=item['preference_boost']) for item in result['items']]
    return {
        'user_id': req.user_id,
        'items': items,
        'model_version': 'preference-v1',
        'cached': False,
        'preference_confidence': result['preference_confidence']
    }

@app.get("/user/{user_id}/profile")
def user_profile(user_id: str):
    return retrieval.get_user_profile(user_id)

if __name__ == "__main__":
    print('testing')
