from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from ragommender.api.models import SearchRequest, RecommendationItem
from ragommender import llm, retrieval
from datetime import datetime, timezone
import json
import os
import threading
import time

# one json line per search when SEARCH_LOG names a file: real searches, which the eval's llm paraphrases of tags only
# stand in for, to check the ranking against once there are some
search_log = os.environ.get('SEARCH_LOG')
log_lock = threading.Lock()

def log_search(req, body, search_ms):
    if not search_log:
        return
    line = json.dumps({
        'time': datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
        'query': req.query, 'k': req.k, 'explain': req.explain, 'rewrite': req.rewrite,
        'search_ms': round(search_ms, 1),
        'model_version': body['model_version'],
        'mapped_tags': body['mapped_tags'],
        'rewritten_query': body['rewritten_query'],
        'llm_used': body['llm_used'],
        'llm_cached': body['llm_cached'],
        'item_ids': [item.item_id for item in body['items']]
    }, ensure_ascii=False)
    # a log that can't be written mustn't fail the search
    try:
        with log_lock, open(search_log, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except OSError as e:
        print(f"warning: couldn't write the search log {search_log}: {e}")

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
    start = time.perf_counter()
    result = retrieval.search(req.query, req.k, req.explain, req.rewrite)
    items = [RecommendationItem(item_id=item['item_id'], title=item['title'], score=item['score'],
                                reason=item['reason'], explain=item.get('explain')) for item in result['items']]
    body = {
        'items': items,
        'model_version': retrieval.model_version(),
        'rewritten_query': result['rewritten_query'],
        'mapped_tags': result['mapped_tags'],
        'llm_used': result['llm_used'],
        'llm_cached': result['llm_cached']
    }
    log_search(req, body, (time.perf_counter() - start) * 1000)
    return body
