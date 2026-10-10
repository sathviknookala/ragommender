from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from models import SearchRequest, RecommendationItem
import llm
import retrieval

@asynccontextmanager
async def lifespan(app):
    yield
    llm.close()

app = FastAPI(lifespan=lifespan)

@app.post("/search")
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=422, detail='query is blank')
    result = retrieval.search(req.query, req.k, req.explain, req.rewrite)
    items = [RecommendationItem(item_id=item['item_id'], title=item['title'], score=item['score'],
                                reason=item['reason'], explain=item.get('explain')) for item in result['items']]
    return {
        'items': items,
        'model_version': retrieval.model_version,
        'rewritten_query': result['rewritten_query'],
        'llm_used': result['llm_used'],
        'llm_cached': result['llm_cached']
    }
