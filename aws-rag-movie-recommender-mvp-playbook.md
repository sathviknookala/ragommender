# AWS RAG Movie Recommender — **MVP Build Playbook**

A cloud-first, retrieval-heavy movie recommendation service that emphasizes **deployment, orchestration, and production polish** over heavy modeling. You'll ship a public API that returns Top‑N recommendations using **enhanced hybrid retrieval** (OpenSearch KNN + BM25 + Self-Attention Query Enhancement), optional **RAG explanations** via Bedrock, and production-grade AWS components (API Gateway, Lambda/ECS, DynamoDB, S3, IaC, CI/CD, monitoring).

---

## 0) TL;DR Checklist (print me ✅)
- [ ] Define API contract (`/recommend`, `/similar`, `/feedback`)
- [ ] Land MovieLens + TMDB metadata in **S3**
- [ ] Generate **item embeddings** (Bedrock Titan **or** MiniLM via SM Processing)
- [ ] **[NEW]** Implement **self-attention query enhancement** layer
- [ ] Stand up **OpenSearch Serverless** KNN + BM25 indices, bulk index items
- [ ] Create **DynamoDB** `users` table with enhanced semantic profiles
- [ ] Implement **FastAPI** app: query enhancement → retrieval → hybrid merge → semantic re-rank → (optional) RAG explanations
- [ ] Deploy via **API Gateway → Lambda (container)** (or ECS Fargate)
- [ ] Add **cache** (DynamoDB TTL), **quotas**, **idempotency**
- [ ] Wire **CloudWatch** metrics/dashboards/alarms + **Synthetics canary**
- [ ] README with metrics (latency, cache hit), architecture diagram, curl examples

---

## 1) Scope & API Contract (½ day)

**Non-goals:** heavy deep models; complex personalization.  
**Goals:** semantic query understanding via lightweight self-attention; enhanced hybrid retrieval.  
**Latency targets:** p95 < 350ms cached; < 900ms cold (allowing +50ms for self-attention inference).

**Endpoints**
```http
POST /recommend
Body: { "user_id": "u123", "k": 20, "explain": false, "query_text": "funny sci-fi movies like Guardians" }
Resp: { "user_id": "u123", "items": [{ "item_id": "...", "title": "...", "score": 0.83, "reason": ["Sci‑Fi","similar to Interstellar"] }], "model_version": "mvp-0", "cached": false, "semantic_analysis": {"confidence": 0.85, "intent": "similarity_based"} }

GET /similar?item_id=i42&k=20
Resp: { "items": [ { "item_id": "...", "title": "...", "score": 0.91 } ] }

POST /feedback
Body: { "user_id": "u123", "item_id": "i42", "event": "click|watch|like|rating", "value": 1 }
Resp: 202
```

**Enhanced Pydantic models**
```python
class RecRequest(BaseModel):
    user_id: str
    k: int = 20
    explain: bool = False
    query_text: str | None = None  # NEW: optional semantic query

class RecItem(BaseModel):
    item_id: str
    title: str
    score: float
    reason: list[str] | None = None

class SemanticAnalysis(BaseModel):  # NEW
    confidence: float
    primary_intent: str
    all_intents: dict[str, float]
    enhanced_score: float | None = None

class RecResponse(BaseModel):
    user_id: str
    items: list[RecItem]
    model_version: str = "mvp-0"
    cached: bool = False
    semantic_analysis: SemanticAnalysis | None = None  # NEW
```
---

## 2) Data & Embeddings (1 day)

**Datasets**
- MovieLens 25M ratings + items
- TMDB metadata (title, overview, genres, year)

**Enhanced S3 layout**
```
s3://<bucket>/raw/movielens/...
s3://<bucket>/raw/tmdb/...
s3://<bucket>/features/items.parquet
s3://<bucket>/features/item_embeddings.parquet
s3://<bucket>/models/self_attention_query_enhancer.pt  # NEW: lightweight attention model
```

**Option A — Bedrock Titan Embeddings (managed)**
- Concatenate: `title + " " + genres + " " + overview` → `item_vector: float[1536]` (example dim; confirm chosen model dim).
- Write `item_id, text, vector` to `item_embeddings.parquet`.

**Option B — MiniLM (SentenceTransformers) via SageMaker Processing**
- Run a one-shot Processing job to compute vectors (`384` dims typical).
- Persist to `features/item_embeddings.parquet`.

> **Tip:** keep the vector dimension **constant** and record it in `.env` and OpenSearch mappings.

---

## 3) **[NEW]** Self-Attention Query Enhancement (½–1 day)

**Lightweight Attention Model Architecture**
```python
# Core self-attention for query semantic understanding
class QuerySelfAttention(nn.Module):
    def __init__(self, embed_dim=384, num_heads=4, dropout=0.1):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.semantic_classifier = nn.Linear(embed_dim, 5)  # 5 intent categories
        
    def forward(self, query_embeddings):
        attn_out, attn_weights = self.self_attention(query_embeddings, query_embeddings, query_embeddings)
        enhanced = self.layer_norm(query_embeddings + attn_out)
        semantic_scores = F.softmax(self.semantic_classifier(enhanced.mean(dim=1)), dim=-1)
        return enhanced, semantic_scores, attn_weights
```

**Semantic Intent Categories**
- `genre_preference`: "horror movies", "comedies"
- `mood_based`: "feel-good movies", "sad films"  
- `descriptive_features`: "great cinematography", "award-winning"
- `temporal_context`: "recent movies", "90s films"
- `similarity_based`: "like Inception", "similar to Blade Runner"

**Query Enhancement Pipeline**
```python
def enhance_query(query_text: str, user_profile: dict) -> tuple[np.ndarray, dict]:
    # 1) Get base embedding
    base_embedding = sentence_model.encode([query_text])
    
    # 2) Apply self-attention
    enhanced_embedding, semantic_scores, _ = attention_model(torch.FloatTensor(base_embedding).unsqueeze(1))
    
    # 3) Analyze semantic intent
    analysis = {
        "confidence": torch.max(semantic_scores).item(),
        "primary_intent": intent_categories[torch.argmax(semantic_scores).item()],
        "all_intents": {cat: score.item() for cat, score in zip(intent_categories, semantic_scores[0])},
        "keywords": extract_keywords(query_text),
        "entities": extract_entities(query_text)
    }
    
    return enhanced_embedding.detach().numpy(), analysis
```

**Model Training Strategy**
- **Phase 1 (MVP)**: Use pre-trained weights from sentence transformers + simple classification head
- **Phase 2**: Fine-tune on MovieLens query patterns (synthetic queries from user interaction data)

**Deployment**
- **Model Size**: ~10-50MB (fits Lambda constraints)
- **Storage**: S3 → Lambda Layer or EFS
- **Inference**: <50ms additional latency
- **Caching**: Cache enhanced embeddings by query hash in DynamoDB

---

## 4) OpenSearch Serverless: Enhanced Hybrid Retrieval (1 day)

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
      "themes": {"type": "keyword"},  # NEW: extracted thematic tags
      "mood_tags": {"type": "keyword"},  # NEW: mood descriptors
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

**Enhanced Hybrid query with semantic adaptation**
```python
def enhanced_hybrid_query(user_vector, user_keywords, query_analysis, knn_k=200):
    # Adaptive k based on query confidence and intent
    confidence = query_analysis["confidence"]
    intent = query_analysis["primary_intent"]
    
    if confidence < 0.6:  # Low confidence → cast wider net
        adaptive_k = min(knn_k * 2, 400)
    elif intent == "similarity_based":  # High precision queries
        adaptive_k = max(knn_k // 2, 50)
    else:
        adaptive_k = knn_k
    
    # Field boosting based on query intent
    if intent == "mood_based":
        fields = ["overview^3", "mood_tags^2", "title^2", "genres"]
    elif intent == "descriptive_features":
        fields = ["overview^2", "themes^2", "title^2", "genres"]
    else:
        fields = ["title^2", "overview", "genres"]
    
    return {
      "size": adaptive_k,
      "query": {
        "bool": {
          "should": [
            {"knn": {"vector": {"vector": user_vector, "k": adaptive_k}}},
            {"multi_match": {
               "query": user_keywords,
               "fields": fields,
               "type": "best_fields"
            }}
          ]
        }
      }
    }
```
> Enhanced weights at re‑rank time: `score = 0.6 * knn_sim + 0.25 * bm25_score + 0.15 * semantic_boost`.

---

## 5) Enhanced Personalization (½–1 day)

**DynamoDB table (users)**
```bash
aws dynamodb create-table   --table-name users   --attribute-definitions AttributeName=user_id,AttributeType=S   --key-schema AttributeName=user_id,KeyType=HASH   --billing-mode PAY_PER_REQUEST
```

**Enhanced user profile schema**
```json
{
  "user_id": "u123",
  "recent_item_ids": ["i42","i108"],
  "genre_counts": {"Sci-Fi": 9, "Drama": 3},
  "user_vector": [0.013, -0.044, ...],
  "semantic_preferences": {
    "dominant_themes": ["space_exploration", "time_travel", "philosophical"],
    "mood_patterns": ["thought_provoking", "visually_stunning"],
    "query_patterns": ["similarity_based", "genre_preference"]
  },
  "attention_enhanced_vector": [...],
  "query_history": [
    {"query": "sci-fi like Blade Runner", "intent": "similarity_based", "timestamp": 1724700000}
  ],
  "updated_at": 1724700000
}
```

**Seeding enhanced users**
- Compute `user_vector = mean(embedding(item) for item in recent_item_ids)`
- Build `user_keywords` from top genres or favorite titles.
- **NEW**: Extract semantic preferences from viewing history using NLP analysis
- **NEW**: Generate `attention_enhanced_vector` by applying query patterns to user preferences

---

## 6) Semantic-Aware Ranking (½–1 day)

**Enhanced heuristic ranker with semantic intelligence**
```python
def enhanced_rerank(cands, user, popularity, query_analysis, recency_boost=0.05):
    # cands: [{item_id, title, knn_sim, bm25_score, year, genres, themes, mood_tags}]
    
    def genre_match(item): 
        return sum(user["genre_counts"].get(g,0) for g in item["genres"])
    
    def semantic_alignment(item, analysis):
        """NEW: Calculate semantic relevance based on query intent"""
        intent = analysis["primary_intent"]
        confidence = analysis["confidence"]
        
        alignment_score = 0.0
        
        # Intent-specific alignment
        if intent == "mood_based":
            # Check mood tag matches
            user_moods = user.get("semantic_preferences", {}).get("mood_patterns", [])
            item_moods = item.get("mood_tags", [])
            alignment_score += len(set(user_moods) & set(item_moods)) * 0.3
            
        elif intent == "similarity_based":
            # Enhanced thematic similarity
            user_themes = user.get("semantic_preferences", {}).get("dominant_themes", [])
            item_themes = item.get("themes", [])
            alignment_score += len(set(user_themes) & set(item_themes)) * 0.4
            
        elif intent == "descriptive_features":
            # Boost items with rich descriptions
            overview_length = len(item.get("overview", ""))
            alignment_score += min(overview_length / 500, 0.3)  # Normalize to 0-0.3
        
        return alignment_score * confidence  # Weight by query confidence
    
    for x in cands:
        # Base scoring (your existing logic)
        base_score = (
          0.6 * x["knn_sim"] +
          0.25 * x["bm25_score"] +
          0.10 * (popularity.get(x["item_id"], 0)) +
          recency_boost * max(0, (x["year"] - 2015)/10.0) +
          0.05 * genre_match(x)
        )
        
        # NEW: Semantic enhancement
        semantic_boost = semantic_alignment(x, query_analysis)
        
        x["score"] = base_score + (0.15 * semantic_boost)
        x["semantic_score"] = semantic_boost  # For debugging/analysis
    
    # Enhanced diversity: penalize near-duplicate themes (MMR-lite)
    seen_themes = set()
    diversified_cands = []
    
    for cand in sorted(cands, key=lambda z: z["score"], reverse=True):
        item_themes = set(cand.get("themes", []))
        overlap = len(item_themes & seen_themes)
        
        # Penalize high thematic overlap
        if overlap > 2:
            cand["score"] *= 0.8
        
        diversified_cands.append(cand)
        seen_themes.update(item_themes)
    
    return sorted(diversified_cands, key=lambda z: z["score"], reverse=True)
```

**Enhanced Cache Strategy**: 
- Cache includes query semantic analysis: `{ pk: "rec#u123#k20#ex0#intent_genre", value: <topN>, semantic_hash: "abc123", ttl: +1800s }`
- Cache hits only when semantic intent matches (prevents inappropriate cache reuse)

---

## 7) (Optional) RAG Explanations via Bedrock (½ day)

**Enhanced prompt skeleton with semantic context**
```
System: You are a concise recommendation explainer with semantic understanding.
User Context: User likes: <top-genres>. Recent favorites: <titles>. Query intent: <semantic_intent> (confidence: <confidence>).
Item: <title> (<year>) — Genres: <...>. Overview: <120-char snippet>. Themes: <themes>. Mood: <mood_tags>.
Semantic Match: This item aligns with user's <intent> preference because <semantic_reasoning>.
Task: In 1 sentence (<= 25 words), explain why this item fits both the user's profile AND their current query intent. Avoid spoilers.
Return only the sentence.
```

**Semantic-aware explanation batching**
```python
def generate_explanations(items, user, query_analysis):
    # Group items by semantic similarity for batch efficiency
    intent = query_analysis["primary_intent"]
    
    # Tailor explanations to query intent
    if intent == "similarity_based":
        focus = "thematic and stylistic similarities"
    elif intent == "mood_based":
        focus = "emotional tone and atmosphere"
    else:
        focus = "genre preferences and viewing history"
    
    # Batch prompt with semantic context
    return batch_bedrock_prompt(items, user, query_analysis, focus)
```

---

## 8) API & Deployment (1–2 days)

**Enhanced FastAPI app (core flow with self-attention)**
```python
from fastapi import FastAPI
from pydantic import BaseModel
import boto3, json, os
from services.query_enhancement import QueryEnhancer  # NEW

app = FastAPI()
query_enhancer = QueryEnhancer()  # NEW: Load self-attention model

@app.post("/recommend", response_model=RecResponse)
def recommend(req: RecRequest):
    # 0) check semantic cache (DynamoDB get with query intent)
    cache_key = f"rec#{req.user_id}#k{req.k}#ex{int(req.explain)}"
    
    # NEW: Add semantic analysis to cache key if query_text provided
    semantic_analysis = None
    if req.query_text:
        enhanced_embedding, semantic_analysis = query_enhancer.enhance_query(req.query_text)
        cache_key += f"#{semantic_analysis['primary_intent']}"
    
    # 1) load user profile (DynamoDB)
    # 2) build retrieval inputs (user_vector, user_keywords, enhanced_embedding)
    # 3) OpenSearch: run enhanced hybrid query -> candidates
    # 4) compute features & enhanced semantic re-rank; filter seen
    # 5) (optional) semantic-aware explanations via Bedrock
    # 6) put cache + emit CloudWatch metrics (including semantic metrics)
    
    return RecResponse(
        user_id=req.user_id,
        items=ranked_items,
        cached=cache_hit,
        semantic_analysis=semantic_analysis  # NEW
    )
```

**Deploy Option A — Lambda (container) [RECOMMENDED]**
- Package FastAPI + Mangum adapter + self-attention model (~50MB total).
- **API Gateway (HTTP API)** → Lambda.
- **Model Loading**: Load from S3/EFS on cold start, cache in memory.
- Pros: simple, cheap for bursty traffic, auto-scaling.

**Deploy Option B — ECS Fargate**
- Containerize FastAPI with Uvicorn/Gunicorn + model.
- ALB → ECS service.
- Pros: steady throughput, consistent latency, easier model management.

**Enhanced IaC**: **AWS CDK** (TS or Py) creates:
- S3 buckets (including model artifacts), DynamoDB (with semantic cache fields), OpenSearch collection & access policy, API Gateway, Lambda/ECS (with ML inference optimizations), IAM roles, CloudWatch dashboard (with semantic metrics), WAF.

---

## 9) Caching, Quotas, Idempotency (½ day)
**Enhanced DynamoDB cache row**
```json
{ 
  "pk": "rec#u123#k20#ex0#genre_preference", 
  "value": "<json>", 
  "semantic_hash": "abc123def",
  "intent": "genre_preference",
  "confidence": 0.85,
  "ttl": 1724703600 
}
```

**Semantic Cache Strategy**
- Cache hits require matching user_id, k, explain AND semantic intent
- Different intents get separate cache entries (prevents inappropriate reuse)
- Lower confidence queries (<0.6) have shorter TTL (900s vs 1800s)

**Quotas**
- Maintain per‑API‑key counters in DynamoDB (rolling window). Return `429` on exceed.
- **NEW**: Separate quotas for semantic-enhanced vs basic queries

**Idempotency (feedback)**
- Compound key: `"fb#u123#i42#ts169..."`. Ignore duplicates within 1 minute.

---

## 10) Enhanced Observability & SLOs (½–1 day)
**Enhanced metrics to emit**
- `API.LatencyMs`, `API.ErrorRate`, `API.RPS`
- `Retrieval.KNNLatencyMs`, `Retrieval.BM25LatencyMs`
- `RankerLatencyMs`, `Cache.HitRatio`
- `Explain.UsagePct`
- **NEW**: `SelfAttention.InferenceLatencyMs`, `SelfAttention.CacheHitRatio`
- **NEW**: `QueryAnalysis.ConfidenceScore`, `QueryAnalysis.IntentDistribution`
- **NEW**: `SemanticBoost.AverageScore`, `SemanticCache.HitRatio`

**Enhanced CloudWatch**
- Dashboard with above metrics + semantic analysis distribution.
- Alarms: p95 latency > 900ms (updated for semantic processing), ErrorRate > 2%, SelfAttention latency > 100ms.

**Synthetics Canary**
- Hit `/recommend` every 5 min with fixed user + semantic queries; assert 200 + K items + semantic_analysis present.
- **NEW**: Test various query intents to ensure semantic enhancement works.

---

## 11) Validation, README & Cost Hygiene (½ day)
**Functional**: `/recommend`, `/similar`, `/feedback` all work with seeded users AND semantic queries.  
**Quality**: Recall@10 better than popularity baseline (+20–30% with semantic enhancement vs +15–25% baseline).  
**Performance**: p95 < 900ms cold, < 350ms cached (including semantic processing).  
**Production**: IaC deploy; dashboards & alarms green; semantic features working; keys enforced.

**Enhanced README must include**
- Architecture diagram showing self-attention flow (Mermaid or PNG)
- Curl examples with semantic queries
- Metrics table (latency p50/p95, cache hit %, semantic analysis accuracy)
- Semantic query examples and expected intents
- Cost notes: Lambda + small OpenSearch + semantic model storage costs, caching effectiveness

**Cost Considerations**
- **Self-attention inference**: ~+10-20% compute cost (offset by better caching through semantic understanding)
- **Enhanced caching**: Better cache hit rates due to semantic intent matching
- **Model storage**: ~$0.50/month for S3 storage of 50MB model

---

## Appendix A — Enhanced Repo Layout
```
repo/
  infra/                 # CDK/Terraform
  services/api/
    app/
      main.py            # FastAPI handlers
      models.py          # Enhanced Pydantic schemas
      retrieval.py       # Enhanced OpenSearch hybrid query
      ranker.py          # Semantic-aware ranking
      cache.py           # Semantic-aware DynamoDB cache
      users.py           # Enhanced DynamoDB user profiles
      explain.py         # Semantic-aware Bedrock explanations
      query_enhancement.py # NEW: Self-attention query enhancement
    models/
      self_attention_model.py # NEW: PyTorch attention model
    Dockerfile
    tests/
      test_semantic_enhancement.py # NEW: Test semantic features
  data_jobs/
    embed_items.py       # Enhanced embedding with thematic tags
    index_opensearch.py  # Enhanced bulk indexer
    train_attention.py   # NEW: Self-attention model training
  model_artifacts/       # NEW: Trained model storage
    query_enhancer.pt
    intent_categories.json
  README.md
```

## Appendix B — Enhanced Environment Variables
```
OPENSEARCH_ENDPOINT=...
OPENSEARCH_INDEX=movies_vector
VECTOR_DIM=384
AWS_REGION=us-east-1
DDB_USERS_TABLE=users
DDB_CACHE_TABLE=rec_cache
BEDROCK_MODEL=anthropic.claude-3-haiku # optional
CACHE_TTL_SECS=1800
SEMANTIC_CACHE_TTL_SECS=900  # NEW: shorter TTL for low confidence queries
SELF_ATTENTION_MODEL_PATH=s3://bucket/models/query_enhancer.pt  # NEW
USE_SEMANTIC_ENHANCEMENT=true  # NEW: feature flag
SEMANTIC_CONFIDENCE_THRESHOLD=0.5  # NEW: minimum confidence for enhancement
MAX_SEMANTIC_CACHE_ENTRIES=10000  # NEW: limit semantic cache size
```

## Appendix C — Enhanced OpenSearch Query from FastAPI
```python
def enhanced_hybrid_search(os_client, user_vector, keywords, query_analysis, k=200):
    # Adaptive parameters based on semantic analysis
    confidence = query_analysis["confidence"]
    intent = query_analysis["primary_intent"]
    
    # Adjust k and field weights based on intent
    if intent == "mood_based":
        fields = ["overview^3", "mood_tags^2", "title^2", "genres"]
        adaptive_k = min(k * 1.5, 300)  # Mood queries benefit from more candidates
    elif intent == "similarity_based":
        fields = ["themes^2", "title^2", "overview", "genres"]
        adaptive_k = k  # Precise queries don't need expansion
    else:
        fields = ["title^2", "overview", "genres"]
        adaptive_k = k
    
    q = {
      "size": int(adaptive_k),
      "query": {
        "bool": {
          "should": [
            {"knn": {"vector": {"vector": user_vector, "k": int(adaptive_k)}}},
            {"multi_match": {"query": keywords, "fields": fields}}
          ]
        }
      }
    }
    r = os_client.search(index=os.environ["OPENSEARCH_INDEX"], body=q)
    
    # Return with semantic scoring
    results = []
    for hit in r["hits"]["hits"]:
        result = hit["_source"] | {
            "knn_sim": hit["_score"], 
            "bm25_score": 0.0,
            "semantic_intent": intent,
            "query_confidence": confidence
        }
        results.append(result)
    
    return results
```

## Appendix D — Enhanced DynamoDB Cache Helpers
```python
def semantic_cache_key(user_id, k, explain, query_analysis=None):
    base_key = f"rec#{user_id}#k{k}#ex{int(explain)}"
    
    if query_analysis:
        intent = query_analysis["primary_intent"]
        confidence_bucket = "high" if query_analysis["confidence"] > 0.7 else "low"
        return f"{base_key}#{intent}#{confidence_bucket}"
    
    return base_key

def should_use_semantic_cache(query_analysis, confidence_threshold=0.5):
    """Decide whether to use semantic caching based on query analysis"""
    if not query_analysis:
        return False
    
    return (
        query_analysis["confidence"] > confidence_threshold and
        query_analysis["primary_intent"] != "unknown"
    )
```

## Appendix E — Cost Guardrails
- Prefer **Lambda** for MVP; ECS/Fargate only if you need sustained throughput.
- **Serverless OpenSearch** with modest capacity; prune fields, tight mappings.
- **Semantic Enhancement**: Use feature flags to control rollout and costs.
- **Smart Caching**: Higher cache hit rates with semantic intent matching.
- **Model Optimization**: Use quantized models if inference becomes expensive.
- Batch embedding/indexing; avoid per-request embedding.
- **Monitoring**: Set CloudWatch alarms for semantic processing costs.

## Appendix F — Self-Attention Model Training Strategy

**Phase 1: MVP (No Training Required)**
```python
# Use pre-trained transformer components
class MVPQueryEnhancer:
    def __init__(self):
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.intent_classifier = pipeline('text-classification', 
                                          model='facebook/bart-large-mnli')
        
    def enhance_query(self, query_text):
        # Simple intent classification using zero-shot
        intents = ["genre preference", "mood based", "similarity based", 
                  "descriptive features", "temporal context"]
        
        results = self.intent_classifier(query_text, intents)
        # Process results and return enhanced embedding + analysis
```

**Phase 2: Fine-tuned Model**
- Collect query interaction data from production
- Generate synthetic query-intent pairs from MovieLens data
- Fine-tune lightweight attention model on movie domain
- A/B test against MVP implementation

**Training Data Strategy**
- **Synthetic Queries**: Generate from user viewing patterns ("Users who watched X,Y,Z might search for...")
- **Intent Labeling**: Use movie metadata to auto-generate intent labels
- **Validation Set**: Manual labeling of diverse query types

This enhanced playbook maintains your production-first approach while adding sophisticated semantic query understanding that will significantly improve recommendation quality and user experience.