from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from ragommender.api.models import SearchRequest, RecommendationItem
from ragommender import llm, retrieval

@asynccontextmanager
async def lifespan(app):
    # load the index and embedding model before the first request
    retrieval.index()
    retrieval.vocab()
    yield
    llm.close()

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    # the lifespan loads the index before the server takes requests, so an answer here means search is ready
    # llm is 'disabled' (LLM_ENABLED=0), 'cooldown' after a failure (search falls back to templates for a while) or
    # 'available' when no call failed lately. the server isn't called, so this stays cheap to poll
    state = 'disabled' if not llm.enabled else 'available' if llm.is_up() else 'cooldown'
    return {'status': 'ok', 'model_version': retrieval.model_version(), 'mapping': retrieval.vocab() is not None,
            'llm': state}

@app.post("/search")
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=422, detail='query is blank')
    result = retrieval.search(req.query, req.k, req.explain, req.rewrite)
    items = [RecommendationItem(item_id=item['item_id'], title=item['title'], score=item['score'],
                                reason=item['reason'], explain=item.get('explain')) for item in result['items']]
    return {
        'items': items,
        'model_version': retrieval.model_version(),
        'rewritten_query': result['rewritten_query'],
        'mapped_tags': result['mapped_tags'],
        'llm_used': result['llm_used'],
        'llm_cached': result['llm_cached']
    }
