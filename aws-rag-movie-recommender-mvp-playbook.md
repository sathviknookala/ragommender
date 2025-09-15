# AWS RAG Movie Recommender — **MVP Build Playbook**

A cloud-first, retrieval-heavy movie recommendation service that emphasizes **deployment, orchestration, and production polish** over heavy modeling. You’ll ship a public API that returns Top‑N recommendations using **hybrid retrieval** (OpenSearch KNN + BM25), optional **RAG explanations** via Bedrock, and production-grade AWS components (API Gateway, Lambda/ECS, DynamoDB, S3, IaC, CI/CD, monitoring).

---

## 0) TL;DR Checklist (print me ✅)
- [ ] Define API contract (`/recommend`, `/similar`, `/feedback`)
- [ ] Land MovieLens + TMDB metadata in **S3**
- [ ] Generate **item embeddings** (Bedrock Titan **or** MiniLM via SM Processing)
- [ ] Stand up **OpenSearch Serverless** KNN + BM25 indices, bulk index items
- [ ] Create **DynamoDB** `users` table; seed a few user profiles (`recent_item_ids`, `genre_counts`, `user_vector`)
- [ ] Implement **FastAPI** app: retrieval → hybrid merge → light re-rank → (optional) RAG explanations
- [ ] Deploy via **API Gateway → Lambda (container)** (or ECS Fargate)
- [ ] Add **cache** (DynamoDB TTL), **quotas**, **idempotency**
- [ ] Wire **CloudWatch** metrics/dashboards/alarms + **Synthetics canary**
- [ ] README with metrics (latency, cache hit), architecture diagram, curl examples

---

## 1) Scope & API Contract (½ day)

**Non-goals:** heavy deep models; complex personalization.  
**Latency targets:** p95 < 300ms cached; < 800ms cold.

**Endpoints**
```http
POST /recommend
Body: { "user_id": "u123", "k": 20, "explain": false }
Resp: { "user_id": "u123", "items": [{ "item_id": "...", "title": "...", "score": 0.83, "reason": ["Sci‑Fi","similar to Interstellar"] }], "model_version": "mvp-0", "cached": false }

GET /similar?item_id=i42&k=20
Resp: { "items": [ { "item_id": "...", "title": "...", "score": 0.91 } ] }

POST /feedback
Body: { "user_id": "u123", "item_id": "i42", "event": "click|watch|like|rating", "value": 1 }
Resp: 202
```

**Pydantic models**
```python
class RecRequest(BaseModel):
    user_id: str
    k: int = 20
    explain: bool = False

class RecItem(BaseModel):
    item_id: str
    title: str
    score: float
    reason: list[str] | None = None

class RecResponse(BaseModel):
    user_id: str
    items: list[RecItem]
    model_version: str = "mvp-0"
    cached: bool = False
```
---

## 2) Data & Embeddings (1 day)

**Datasets**
- MovieLens 25M ratings + items
- TMDB metadata (title, overview, genres, year)

**S3 layout**
```
s3://<bucket>/raw/movielens/...
s3://<bucket>/raw/tmdb/...
s3://<bucket>/features/items.parquet
s3://<bucket>/features/item_embeddings.parquet
```

**Option A — Bedrock Titan Embeddings (managed)**
- Concatenate: `title + " " + genres + " " + overview` → `item_vector: float[1536]` (example dim; confirm chosen model dim).
- Write `item_id, text, vector` to `item_embeddings.parquet`.

**Option B — MiniLM (SentenceTransformers) via SageMaker Processing**
- Run a one-shot Processing job to compute vectors (`384` dims typical).
- Persist to `features/item_embeddings.parquet`.

> **Tip:** keep the vector dimension **constant** and record it in `.env` and OpenSearch mappings.

---

## 3) OpenSearch Serverless: Hybrid Retrieval (1 day)

**Create collection & indices (KNN + BM25)**

**KNN index mapping (Python)**
```python
from opensearchpy import OpenSearch
client = OpenSearch(<endpoint>, http_auth=(user, pwd), use_ssl=True, verify_certs=True)

index_name = "movies_vector"
dim = 384  # or 1536 if Titan
body = {
  "settings": { "index": { "knn": True } },
  "mappings": {
    "properties": {
      "item_id": {"type": "keyword"},
      "title": {"type": "text"},
      "overview": {"type": "text"},
      "genres": {"type": "keyword"},
      "year": {"type": "integer"},
      "vector": {
        "type": "knn_vector",
        "dimension": dim,
        "method": {"name": "hnsw","space_type": "cosinesimil","engine": "nmslib"}
      }
    }
  }
}
client.indices.create(index=index_name, body=body)
```

**BM25 text index**  
You can re‑use the same index (text fields above) or create a separate `movies_text` index tuned for BM25.

**Bulk ingest (Python)**
```python
from opensearchpy.helpers import bulk
def to_actions(rows):
    for r in rows:  # item_id, title, overview, genres(list), year, vector(list)
        yield {"_index": "movies_vector", "_id": r["item_id"], "_source": r}

bulk(client, to_actions(rows))
```

**Hybrid query (KNN + BM25)**
```python
def hybrid_query(user_vector, user_keywords, knn_k=200):
    return {
      "size": knn_k,
      "query": {
        "bool": {
          "should": [
            {"knn": {"vector": {"vector": user_vector, "k": knn_k}}},
            {"multi_match": {
               "query": user_keywords,
               "fields": ["title^2","overview","genres"],
               "type": "best_fields"
            }}
          ]
        }
      }
    }
```
> Tune weights at re‑rank time: `score = 0.7 * knn_sim + 0.3 * bm25_score`.

---

## 4) Minimal Personalization (½–1 day)

**DynamoDB table (users)**
```bash
aws dynamodb create-table   --table-name users   --attribute-definitions AttributeName=user_id,AttributeType=S   --key-schema AttributeName=user_id,KeyType=HASH   --billing-mode PAY_PER_REQUEST
```

**User profile schema**
```json
{
  "user_id": "u123",
  "recent_item_ids": ["i42","i108"],
  "genre_counts": {"Sci-Fi": 9, "Drama": 3},
  "user_vector": [0.013, -0.044, ...],
  "updated_at": 1724700000
}
```

**Seeding users**
- Compute `user_vector = mean(embedding(item) for item in recent_item_ids)`
- Build `user_keywords` from top genres or favorite titles.

---

## 5) Lightweight Ranking (½ day)

**Heuristic ranker (fast, no training)**
```python
def rerank(cands, user, popularity, recency_boost=0.05):
    # cands: [{item_id, title, knn_sim, bm25_score, year, genres}]
    def genre_match(item): 
        return sum(user["genre_counts"].get(g,0) for g in item["genres"])
    for x in cands:
        x["score"] = (
          0.7 * x["knn_sim"] +
          0.3 * x["bm25_score"] +
          0.10 * (popularity.get(x["item_id"], 0)) +
          recency_boost * max(0, (x["year"] - 2015)/10.0) +
          0.05 * genre_match(x)
        )
    # diversity: penalize near-duplicate genres (MMR-lite)
    return sorted(cands, key=lambda z: z["score"], reverse=True)
```

**Cache**: DynamoDB item `{ pk: "rec#u123#k20#ex0", value: <topN>, ttl: +1800s }`

---

## 6) (Optional) RAG Explanations via Bedrock (½ day)

**Prompt skeleton**
```
System: You are a concise recommendation explainer.
User: User likes: <top-genres>. Recent favorites: <titles>.
Item: <title> (<year>) — Genres: <...>. Overview: <120-char snippet>.
Task: In 1 sentence (<= 22 words), explain why this item fits the user's tastes. Avoid spoilers.
Return only the sentence.
```

Make it **opt‑in** via `explain=true` to protect latency. Batch prompt top‑N to reduce calls.

---

## 7) API & Deployment (1–2 days)

**FastAPI app (core flow)**
```python
from fastapi import FastAPI
from pydantic import BaseModel
import boto3, json, os

app = FastAPI()

@app.post("/recommend", response_model=RecResponse)
def recommend(req: RecRequest):
    # 0) check cache (DynamoDB get)
    # 1) load user profile (DynamoDB)
    # 2) build retrieval inputs (user_vector, user_keywords)
    # 3) OpenSearch: run hybrid query -> candidates
    # 4) compute features & rerank; filter seen
    # 5) (optional) explanations via Bedrock
    # 6) put cache + emit CloudWatch metrics
    return response
```

**Deploy Option A — Lambda (container)**
- Package FastAPI + Mangum adapter.
- **API Gateway (HTTP API)** → Lambda.
- Pros: simple, cheap for bursty traffic.

**Deploy Option B — ECS Fargate**
- Containerize FastAPI with Uvicorn/Gunicorn.
- ALB → ECS service.
- Pros: steady throughput, consistent latency.

**IaC**: **AWS CDK** (TS or Py) creates:
- S3 buckets, DynamoDB, OpenSearch collection & access policy, API Gateway, Lambda/ECS, IAM roles, CloudWatch dashboard, WAF.

---

## 8) Caching, Quotas, Idempotency (½ day)
**DynamoDB cache row**
```json
{ "pk": "rec#u123#k20#ex0", "value": "<json>", "ttl": 1724703600 }
```
**Quotas**
- Maintain per‑API‑key counters in DynamoDB (rolling window). Return `429` on exceed.

**Idempotency (feedback)**
- Compound key: `"fb#u123#i42#ts169..."`. Ignore duplicates within 1 minute.

---

## 9) Observability & SLOs (½–1 day)
**Metrics to emit**
- `API.LatencyMs`, `API.ErrorRate`, `API.RPS`
- `Retrieval.KNNLatencyMs`, `Retrieval.BM25LatencyMs`
- `RankerLatencyMs`, `Cache.HitRatio`
- `Explain.UsagePct`

**CloudWatch**
- Dashboard with above metrics.
- Alarms: p95 latency > 800ms (5m, 3 datapoints), ErrorRate > 2%.

**Synthetics Canary**
- Hit `/recommend` every 5 min with a fixed user; assert 200 + K items.

---

## 10) Validation, README & Cost Hygiene (½ day)
**Functional**: `/recommend`, `/similar`, `/feedback` all work with seeded users.  
**Quality**: Recall@10 better than popularity baseline on a small split (+15–25%).  
**Performance**: p95 < 800ms cold, < 300ms cached.  
**Production**: IaC deploy; dashboards & alarms green; keys enforced.

**README must include**
- Architecture diagram (Mermaid or PNG)
- Curl examples
- Metrics table (latency p50/p95, cache hit %)
- Cost notes: small OpenSearch, Lambda over ECS if bursty, caching, shut down idle endpoints.

---

## Appendix A — Minimal Repo Layout
```
repo/
  infra/                 # CDK/Terraform
  services/api/
    app/
      main.py            # FastAPI handlers
      models.py          # Pydantic schemas
      retrieval.py       # OpenSearch hybrid query
      ranker.py          # heuristic ranking
      cache.py           # DynamoDB cache
      users.py           # DynamoDB user profiles
      explain.py         # Bedrock (optional)
    Dockerfile
    tests/
  data_jobs/
    embed_items.py       # Bedrock or MiniLM -> item_embeddings.parquet
    index_opensearch.py  # bulk indexer
  README.md
```

## Appendix B — Environment Variables
```
OPENSEARCH_ENDPOINT=...
OPENSEARCH_INDEX=movies_vector
VECTOR_DIM=384
AWS_REGION=us-east-1
DDB_USERS_TABLE=users
DDB_CACHE_TABLE=rec_cache
BEDROCK_MODEL=anthropic.claude-3-haiku # optional
CACHE_TTL_SECS=1800
```

## Appendix C — Example: OpenSearch Query from FastAPI
```python
def hybrid_search(os_client, user_vector, keywords, k=200):
    q = {
      "size": k,
      "query": {
        "bool": {
          "should": [
            {"knn": {"vector": {"vector": user_vector, "k": k}}},
            {"multi_match": {"query": keywords, "fields": ["title^2","overview","genres"]}}
          ]
        }
      }
    }
    r = os_client.search(index=os.environ["OPENSEARCH_INDEX"], body=q)
    return [hit["_source"] | {"knn_sim": hit["_score"], "bm25_score": 0.0} for hit in r["hits"]["hits"]]
```

## Appendix D — DynamoDB Cache Helpers
```python
def cache_key(user_id, k, explain):
    return f"rec#{user_id}#k{k}#ex{int(explain)}"
```

## Appendix E — Cost Guardrails
- Prefer **Lambda** for MVP; ECS/Fargate only if you need sustained throughput.
- **Serverless OpenSearch** with modest capacity; prune fields, tight mappings.
- Aggressive caching (same input → same output).
- Batch embedding/indexing; avoid per-request embedding.
