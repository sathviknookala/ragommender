# AWS Intelligent Preference-Driven Movie Recommender — **MVP Build Playbook**

A cloud-first, preference-learning movie recommendation service that emphasizes **swipe-based preference collection** and **transformer-based intelligent embedding fusion**. You'll ship a public API that returns Top‑N recommendations using **context-aware hybrid retrieval** (fine-tuned transformer for query+preference fusion → ChromaDB/OpenSearch KNN + BM25), optional **RAG explanations** via Bedrock, and production-grade AWS components (API Gateway, Lambda/ECS, DynamoDB, S3, IaC, CI/CD, monitoring).

---

## 0) TL;DR Checklist (print me ✅)
- [ ] Define API contract (`/survey/start`, `/swipe`, `/search`, `/user/profile`)
- [ ] Land MovieLens + TMDB metadata in **S3**
- [ ] Generate **item embeddings** (Bedrock Titan **or** MiniLM via SM Processing)
- [ ] **[NEW]** Fine-tune **pre-trained transformer** for intelligent query+preference fusion
- [ ] Implement **swipe-based preference collection** system
- [ ] Stand up **ChromaDB/OpenSearch** KNN + BM25 indices, bulk index items
- [ ] Create **DynamoDB** tables (`user_swipes`, `user_preferences`, `enhanced_embeddings_cache`)
- [ ] Implement **unified FastAPI** service: swipe collection → transformer fusion → context-aware hybrid search → ranking
- [ ] Deploy via **API Gateway → Lambda (container)** (or ECS Fargate) + **SageMaker endpoint** for transformer inference
- [ ] Add **intelligent embedding cache** (DynamoDB TTL), **quotas**, **idempotency**
- [ ] Wire **CloudWatch** metrics/dashboards/alarms + **Synthetics canary**
- [ ] README with metrics (latency, cache hit, transformer accuracy), architecture diagram, curl examples

---

## 1) Scope & API Contract (½ day)

**Core Strategy:** Swipe-based preference learning combined with transformer-based intelligent embedding fusion that creates context-aware, personalized search representations.
**Goals:** Rich user preference modeling through swipes; fine-tuned transformer for query+preference fusion; intelligent context-aware hybrid search; production-ready deployment.  
**Latency targets:** p95 < 300ms cached; < 800ms cold (leveraging intelligent embedding cache and optimized transformer inference).

**Endpoints**
```http
POST /survey/start
Body: { "user_id": "u123", "survey_size": 20 }
Resp: { "user_id": "u123", "movies": [{ "item_id": "...", "title": "...", "genres": [...], "year": 2020 }], "session_id": "survey_001" }

POST /swipe
Body: { "user_id": "u123", "item_id": "456", "direction": "like|dislike", "session_id": "survey_001" }
Resp: { "status": "recorded", "preferences_updated": true, "swipe_count": 15 }

POST /search
Body: { "user_id": "u123", "query": "funny sci-fi movies", "k": 20, "explain": false }
Resp: { "user_id": "u123", "items": [{ "item_id": "...", "title": "...", "score": 0.83, "preference_boost": 0.2, "reason": ["Matches your sci-fi preferences"] }], "model_version": "preference-v1", "cached": false, "preference_confidence": 0.78 }

GET /user/{user_id}/profile
Resp: { "user_id": "u123", "swipe_count": 25, "preference_confidence": 0.78, "top_genres": ["Sci-Fi", "Action"], "learned_weights": {"vector": 0.65, "bm25": 0.35, "preference": 0.4} }
```

**Enhanced Pydantic models**
```python
class SwipeEvent(BaseModel):
    user_id: str
    item_id: str
    direction: Literal["like", "dislike"]
    session_id: str
    timestamp: Optional[int] = None

class SearchRequest(BaseModel):
    user_id: str
    query: str
    k: int = 20
    explain: bool = False

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
```

---

## 2) Data & Embeddings (1 day)

**Datasets**
- MovieLens 25M ratings + items
- TMDB metadata (title, overview, genres, year, tags)

**Enhanced S3 layout**
```
s3://<bucket>/raw/movielens/...
s3://<bucket>/raw/tmdb/...
s3://<bucket>/features/items.parquet
s3://<bucket>/features/item_embeddings.parquet
s3://<bucket>/features/survey_movies.json  # NEW: Curated movies for surveys
```

**Option A — Bedrock Titan Embeddings (managed)**
- Concatenate: `title + " " + genres + " " + overview + " " + tags` → `item_vector: float[1536]`.
- Write `item_id, text, vector, genres, tags, year` to `item_embeddings.parquet`.

**Option B — MiniLM (SentenceTransformers) via SageMaker Processing**
- Run a one-shot Processing job to compute vectors (`384` dims typical).
- Persist to `features/item_embeddings.parquet`.

**Survey Movie Curation**
- Select diverse representative movies across genres (Action, Comedy, Drama, Horror, Sci-Fi, Romance)
- Balance popular (broad appeal) vs niche (preference signal) movies
- Include movies from different eras (1980s-2020s)
- Store curated survey sets in S3 for consistent user experience

---

## 3) Swipe Data Collection & Storage (1 day)

**DynamoDB Tables**

**`user_swipes` table:**
```json
{
  "PK": "user#123",
  "SK": "swipe#1640995200#item456", 
  "user_id": "123",
  "item_id": "456",
  "swipe_direction": "like|dislike", 
  "title": "The Matrix",
  "genres": ["Action", "Sci-Fi"],
  "tags": ["cyberpunk", "philosophy"],
  "year": 1999,
  "timestamp": 1640995200,
  "session_id": "survey_001",
  "ttl": 1672531200  // Optional: 1 year retention
}
```

**`user_preferences` table:**
```json
{
  "user_id": "123",
  "preference_vector": [0.1, -0.3, 0.8, ...], // 384-dim learned preference embedding
  "genre_preferences": {
    "Action": {"like_count": 15, "dislike_count": 2, "weight": 0.85},
    "Romance": {"like_count": 3, "dislike_count": 8, "weight": -0.25}
  },
  "tag_preferences": {
    "cyberpunk": 0.9,
    "romantic_comedy": -0.4,
    "time_travel": 0.7
  },
  "learned_weights": {
    "vector_weight": 0.65,
    "bm25_weight": 0.35,
    "preference_boost": 0.4
  },
  "confidence_score": 0.78, // how confident we are in this profile
  "last_updated": 1640995200,
  "swipe_count": 25,
  "feature_preferences": {
    "year_bias": 0.2,  // prefers newer movies
    "popularity_bias": 0.1  // slight preference for popular movies
  }
}
```

---

## 4) Transformer-Based Intelligent Embedding Fusion (2-3 days)

**Core Architecture:** Fine-tuned pre-trained transformer that fuses multiple input streams (query + swipe data + user preferences) into context-rich, intelligent embeddings for personalized search.

**Key Innovation:** Instead of training a transformer from scratch or processing embeddings synchronously, we fine-tune an existing transformer (BERT/RoBERTa) to understand how to intelligently combine user queries with their preference context.

### **Architecture Overview**
```
Multiple Input Streams → Pre-trained Transformer (fine-tuned) → Context-Rich Intelligent Embedding → Hybrid Search
```

**Input Stream Fusion Strategy:**
- **Query Representation**: User's explicit search query ("funny sci-fi movies")
- **Preference Context**: Summarized swipe history and learned preferences
- **User Context**: Temporal patterns, search history, contextual signals

### **Fine-tuning Approach**

**Option A: Token-Based Fusion**
```
Input: [CLS] [QUERY] funny sci-fi [SEP] [LIKES] Matrix Interstellar [SEP] [DISLIKES] romance drama [SEP]
Output: Enhanced query embedding from [CLS] token
```

**Option B: Embedding-Level Fusion**
```
Input: Query Embedding (384d) + Preference Summary (384d) + Context (384d)
Process: Transformer attention layers fuse representations
Output: Context-aware intelligent embedding (384d)
```

### **Why This Architecture Works**

**Leverages Pre-training:**
- Sophisticated attention mechanisms already learned
- Understanding of text relationships and context
- Transfer learning from massive language model training

**Fine-tuning Teaches:**
- Movie domain-specific preference relationships
- How user behavior patterns relate to search intent
- Personalization logic for different query types

### **Training Strategy**

**Self-Supervised Learning on Existing Data:**
```python
# Training Examples from Your Swipe Data
Input: [User query at time T] + [User swipe history before time T] + [User context]
Target: [Movies user engaged with after that query]
Loss: ContrastiveLoss(enhanced_embedding, positive_movies, negative_movies)
```

**Key Benefits:**
- **No manual labeling required** - uses existing swipe engagement data
- **Continuous improvement** - can retrain as more data accumulates
- **Domain-specific** - learns movie recommendation patterns, not general language

### **Intelligent Embedding Properties**

**Context-Aware:**
- Same query produces different embeddings for different users
- User A: "action movies" → Marvel/superhero semantics
- User B: "action movies" → thriller/John Wick semantics

**Temporal Intelligence:**
- Recent preferences weighted more heavily than historical
- Seasonal preference adaptation
- Query context understanding (searching for others vs self)

**Multi-Modal Understanding:**
- Combines explicit query intent with implicit preference signals
- Balances exploration (new content) vs exploitation (known preferences)
- Understands preference contradictions and evolution

### **AWS Deployment Architecture**

**Training Infrastructure:**
- **SageMaker Training Jobs**: Fine-tune transformer on GPU instances
- **Model Registry**: Version and manage fine-tuned models
- **Batch Transform**: Generate embeddings for popular query/user combinations

**Inference Infrastructure:**
- **SageMaker Endpoint**: Real-time transformer inference for new queries
- **Lambda Integration**: Lightweight service calls to transformer endpoint
- **DynamoDB Cache**: Store frequently requested intelligent embeddings

**Caching Strategy:**
```json
{
  "cache_key": "enhanced_embed#u123#query_hash_abc#pref_hash_xyz",
  "enhanced_embedding": [0.1, -0.2, 0.8, ...],
  "confidence_score": 0.85,
  "preference_influence": 0.4,
  "created_at": 1640995200,
  "ttl": 1641081600
}
```

### **Performance Optimization**

**Inference Latency:**
- **Cold queries**: 100-200ms (transformer inference)
- **Cached queries**: 10-20ms (DynamoDB lookup)
- **Batch processing**: Generate embeddings for popular combinations offline

**Model Efficiency:**
- **Distillation**: Create smaller, faster models from fine-tuned transformer
- **Quantization**: Reduce model size for edge deployment
- **ONNX optimization**: Optimize inference performance

**Resource Management:**
- **Auto-scaling**: SageMaker endpoints scale with demand
- **Cost optimization**: Use Spot instances for training, reserved for inference
- **A/B testing**: Compare different fusion strategies with traffic splitting

---

## 5) Enhanced Preference Learning Engine (1-2 days)

**Core Preference Learning Algorithm**

**Functional Components:**
1. **Swipe Data Aggregation**: Collect and process user's like/dislike history
2. **Preference Vector Computation**: Transform swipes into semantic preference embedding
3. **Genre/Tag Preference Calculation**: Statistical analysis of swipe patterns
4. **Adaptive Weight Learning**: Learn personalized search method weights
5. **Confidence Scoring**: Assess reliability of learned preferences

**Key Algorithm Strategies:**
```python
# Preference Vector Computation (3 approaches to test)
def compute_preference_vector(likes, dislikes):
    # Approach 1: Simple centroid difference (fast, interpretable)
    if likes:
        like_embeddings = [get_item_embedding(item) for item in likes]
        like_centroid = np.mean(like_embeddings, axis=0)
    else:
        like_centroid = np.zeros(384)
        
    if dislikes:
        dislike_embeddings = [get_item_embedding(item) for item in dislikes]
        dislike_centroid = np.mean(dislike_embeddings, axis=0)
    else:
        dislike_centroid = np.zeros(384)
        
    # Preference vector = likes - weighted_dislikes
    preference_vector = like_centroid - 0.5 * dislike_centroid
    return preference_vector / np.linalg.norm(preference_vector)
    
    # Approach 2: Recency-weighted averaging
    # Approach 3: Contrastive learning approach
```

**Update Strategy Options:**
- **Real-time**: Update preferences immediately after each swipe (< 50 swipes)
- **Batch**: Hourly/daily updates for power users (> 50 swipes)
- **Hybrid**: Real-time for first 25 swipes, then batch updates

---

## 5) ChromaDB/OpenSearch Hybrid Retrieval (1 day)

**Create collection & indices (KNN + BM25)**

**ChromaDB Setup (Alternative A)**
```python
import chromadb
from rank_bm25 import BM25Okapi

client = chromadb.PersistentClient()
collection = client.get_or_create_collection(
    name="movies",
    metadata={"hnsw:space": "cosine"}
)

# Store movie embeddings with metadata
collection.add(
    ids=movie_ids,
    embeddings=movie_embeddings,
    documents=movie_descriptions,
    metadatas=movie_metadata  # genres, year, tags
)

# Separate BM25 index for text search
bm25_corpus = [preprocess_movie_text(movie) for movie in movies]
bm25_index = BM25Okapi(bm25_corpus)
```

**OpenSearch Setup (Alternative B)**
```python
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
      "tags": {"type": "keyword"},
      "year": {"type": "integer"},
      "vector": {
        "type": "knn_vector",
        "dimension": dim,
        "method": {"name": "hnsw","space_type": "cosinesimil","engine": "nmslib"}
      }
    }
  }
}
```

---

## 6) Context-Aware Hybrid Search with Transformer Fusion (1-2 days)

**Enhanced Search Service Architecture**
```python
class IntelligentContextAwareRetrieval:
    def __init__(self):
        self.collection = chromadb_collection  # or opensearch_client
        self.bm25_index = BM25Okapi(corpus)
        self.transformer_endpoint = SageMakerTransformerEndpoint()
        self.embedding_cache = DynamoDBCache()
        self.preference_engine = PreferenceLearningEngine()
        
    async def intelligent_search(self, user_id: str, query: str, k: int):
        # 1. Get intelligent enhanced embedding
        enhanced_embedding = await self.get_intelligent_query_embedding(
            user_id, query
        )
        
        # 2. Execute hybrid search with intelligent embedding
        vector_results = await self.vector_search(enhanced_embedding, k*3)
        bm25_results = await self.bm25_search(query, k*3)  # Original query for text search
        
        # 3. Context-aware ranking with transformer insights
        final_results = await self.context_aware_ranking(
            vector_results, bm25_results, enhanced_embedding, k
        )
        
        return final_results
        
    async def get_intelligent_query_embedding(self, user_id: str, query: str):
        # 1. Check cache first
        cache_key = self.generate_cache_key(user_id, query)
        cached_embedding = await self.embedding_cache.get(cache_key)
        
        if cached_embedding and not self.is_embedding_stale(cached_embedding):
            return cached_embedding['enhanced_embedding']
        
        # 2. Generate new intelligent embedding via transformer
        user_context = await self.get_user_context(user_id)
        enhanced_embedding = await self.transformer_endpoint.predict({
            'query': query,
            'user_preferences': user_context['preferences'],
            'swipe_history': user_context['recent_swipes'],
            'contextual_signals': user_context['context']
        })
        
        # 3. Cache the result
        await self.embedding_cache.set(cache_key, {
            'enhanced_embedding': enhanced_embedding,
            'confidence_score': enhanced_embedding.confidence,
            'preference_influence': enhanced_embedding.preference_weight,
            'created_at': time.time()
        })
        
        return enhanced_embedding
```

**Preference Enhancement Strategies:**
1. **Vector Query Enhancement**: Blend user query embedding with preference vector
2. **Text Query Enhancement**: Add preferred genre/tag terms to search query
3. **Adaptive Retrieval Scope**: Adjust candidate pool size based on preference confidence
4. **Personalized Field Weighting**: Weight title vs overview vs genres based on user patterns

---

## 7) Preference-Aware Ranking System (1 day)

**Enhanced ranking with learned user preferences**
```python
def preference_weighted_ranking(self, vector_results, bm25_results, user_prefs, k):
    combined_items = self.merge_results(vector_results, bm25_results)
    
    for item in combined_items:
        # Base hybrid score with learned weights
        base_score = (
            user_prefs.learned_weights.vector_weight * item['vector_score'] + 
            user_prefs.learned_weights.bm25_weight * item['bm25_score']
        )
        
        # Preference boost calculation
        preference_boost = self.calculate_preference_alignment(item, user_prefs)
        
        # Final score
        item['final_score'] = base_score + (user_prefs.learned_weights.preference_boost * preference_boost)
        item['preference_boost'] = preference_boost  # For transparency
    
    # Apply diversity and return top-k
    return self.apply_diversity_ranking(combined_items, k)

def calculate_preference_alignment(self, item, user_prefs):
    alignment = 0.0
    
    # Genre alignment
    for genre in item.get('genres', []):
        if genre in user_prefs.genre_preferences:
            alignment += user_prefs.genre_preferences[genre]['weight']
    
    # Tag alignment  
    for tag in item.get('tags', []):
        if tag in user_prefs.tag_preferences:
            alignment += user_prefs.tag_preferences[tag]
    
    # Vector similarity to preference vector
    if user_prefs.preference_vector:
        item_embedding = self.get_item_embedding(item)
        vector_similarity = np.dot(user_prefs.preference_vector, item_embedding)
        alignment += vector_similarity * 0.5
    
    return np.tanh(alignment)  # Normalize between -1 and 1
```

---

## 8) Cold Start & Survey Strategy (½ day)

**Survey Movie Curation Algorithm**
```python
def get_survey_movies(user_id: str, survey_size: int = 20):
    # Strategy: Maximize preference signal while minimizing survey length
    movies = []
    
    # Core genres (3-4 movies each for strong signal)
    core_genres = ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Horror']
    for genre in core_genres:
        movies.extend(get_genre_representatives(genre, 3))
    
    # Era diversity (1-2 per decade)
    movies.extend(get_era_representatives(['1990s', '2000s', '2010s'], 1))
    
    # Popular baseline movies (everyone has an opinion)
    movies.extend(get_universally_known_movies(3))
    
    return random.sample(movies, survey_size)

def get_genre_representatives(genre: str, count: int):
    # Balance popular + critically acclaimed + niche within genre
    popular = get_popular_movies_by_genre(genre, count//2)
    acclaimed = get_acclaimed_movies_by_genre(genre, count//2)
    return popular + acclaimed
```

**Cold Start Fallback Strategy**
- Users with < 5 swipes: Generic popularity-based ranking
- Users with 5-15 swipes: Lightweight preference hints (genre boosting only)
- Users with 15+ swipes: Full preference-informed search

---

## 9) API & Deployment with Transformer Integration (2-3 days)

**Enhanced FastAPI Service with Transformer Integration**
```python
from fastapi import FastAPI
from models import SwipeEvent, SearchRequest, SurveyRequest
import asyncio

app = FastAPI()
intelligent_retrieval = IntelligentContextAwareRetrieval()

@app.post("/survey/start")
async def start_survey(request: SurveyRequest):
    survey_movies = await intelligent_retrieval.get_survey_movies(request.user_id, request.survey_size)
    return {"movies": survey_movies, "session_id": generate_session_id()}

@app.post("/swipe")
async def record_swipe(swipe: SwipeEvent):
    # Store swipe and invalidate relevant caches
    await intelligent_retrieval.store_swipe_with_cache_invalidation(swipe)
    
    # Update preferences and trigger transformer cache refresh if needed
    if intelligent_retrieval.should_update_immediately(swipe.user_id):
        await intelligent_retrieval.update_user_preferences(swipe.user_id)
        # Asynchronously refresh popular query embeddings for this user
        asyncio.create_task(intelligent_retrieval.refresh_user_embedding_cache(swipe.user_id))
    
    swipe_count = await intelligent_retrieval.get_user_swipe_count(swipe.user_id)
    return {"status": "recorded", "swipe_count": swipe_count, "transformer_cache_refreshed": True}

@app.post("/search")
async def intelligent_search(request: SearchRequest):
    # Use transformer-enhanced intelligent search
    results = await intelligent_retrieval.intelligent_search(
        request.user_id, request.query, request.k
    )
    
    # Include transformer insights in response
    return {
        "results": results,
        "intelligence_metadata": {
            "embedding_cached": results.was_cached,
            "preference_influence": results.preference_weight,
            "confidence_score": results.confidence,
            "transformer_latency_ms": results.transformer_time
        }
    }

@app.get("/user/{user_id}/profile")
async def get_user_profile(user_id: str):
    profile = await intelligent_retrieval.get_user_profile_summary(user_id)
    # Add transformer-specific metrics
    profile["transformer_metrics"] = await intelligent_retrieval.get_user_transformer_stats(user_id)
    return profile
```

**Deployment Architecture: Hybrid Lambda + SageMaker**

**Core FastAPI Service (Lambda Container)**
- Handles API requests and orchestration
- Lightweight operations (caching, data retrieval)
- Calls SageMaker endpoint for transformer inference
- Auto-scales based on request volume

**Transformer Service (SageMaker Endpoint)**
- Dedicated GPU instances for transformer inference
- Auto-scaling based on embedding generation demand
- Model versioning and A/B testing capabilities
- Optimized for batch processing popular queries

**Background Services (ECS/Lambda)**
- Batch embedding generation for popular query/user combinations
- Model retraining pipelines
- Cache warming and maintenance

**Deploy Option A — Lambda (container) [RECOMMENDED]**
- Single FastAPI service with all components (~100-200MB total)
- **API Gateway (HTTP API)** → Lambda
- **Model Loading**: SentenceTransformer + BM25 loaded once on cold start
- Pros: simple, cost-effective for variable traffic, auto-scaling

**Deploy Option B — ECS Fargate**
- Containerized FastAPI with consistent performance
- ALB → ECS service
- Pros: predictable latency, easier debugging, better for sustained load

---

## 10) Preference-Aware Caching (½ day)

**Enhanced DynamoDB cache strategy**
```json
{ 
  "pk": "search#u123#query_hash_abc123", 
  "results": "<json>", 
  "preference_hash": "pref_xyz789",  // Hash of user preferences at cache time
  "confidence_score": 0.78,
  "cached_at": 1640995200,
  "ttl": 1724703600 
}
```

**Cache Strategy:**
- Cache hits require matching user_id AND current preference_hash
- Higher confidence preferences get longer TTL (1800s vs 900s)
- Generic searches (cold start users) cached separately with longer TTL

**Smart Cache Invalidation:**
- Invalidate user cache when preferences update significantly
- Partial cache invalidation for minor preference updates
- Global cache warming for popular queries

---

## 11) Enhanced Observability & Transformer Metrics (½–1 day)

**Key Metrics to Track:**
- **API Performance**: `API.LatencyMs`, `API.ErrorRate`, `API.RPS`
- **Transformer Intelligence**: `Transformer.InferenceLatencyMs`, `Transformer.CacheHitRatio`, `Transformer.ConfidenceScore`
- **Embedding Quality**: `Embedding.PreferenceInfluence`, `Embedding.ContextualRelevance`, `Embedding.UserSatisfaction`
- **Preference Learning**: `PreferenceLearning.UpdateLatencyMs`, `PreferenceLearning.ConfidenceScore`
- **Search Quality**: `Search.IntelligentHitRatio`, `Search.DiversityScore`, `Search.PersonalizationLift`
- **User Engagement**: `Survey.CompletionRate`, `Swipe.ConversionRate`, `Search.ClickThroughRate`
- **Cache Performance**: `Cache.HitRatio`, `Cache.IntelligentEmbeddingCacheRatio`, `Cache.TransformerCacheEfficiency`
- **System Health**: `Retrieval.VectorLatencyMs`, `Retrieval.BM25LatencyMs`, `SageMaker.EndpointLatency`

**CloudWatch Dashboard:**
- **Transformer Performance**: Real-time inference latency, cache hit rates, confidence distributions
- **Intelligence Quality**: Preference influence scores, contextual relevance metrics
- **Search Performance**: Comparison of intelligent vs generic search effectiveness
- **User Engagement**: Survey completion, swipe patterns, search satisfaction
- **System Health**: SageMaker endpoint health, embedding cache performance
- **Cost Tracking**: Transformer inference costs, cache efficiency ROI

**Enhanced Alerts:**
- P95 latency > 800ms (including transformer overhead)
- Transformer inference failures > 3%
- Intelligent embedding cache hit rate < 60%
- SageMaker endpoint errors > 1%
- Preference learning failures > 5%
- Survey completion rate < 60%
- Search error rate > 2%

---

## 12) Validation & Success Metrics (½ day)

**Functional Validation:**
- All API endpoints work with seeded users
- Preference learning works with simulated swipe data
- Search results improve with more preference data
- Cache performance meets targets

**Quality Metrics:**
- **Recommendation Quality**: A/B test preference-aware vs generic search
- **User Engagement**: Click-through rates, session duration
- **Preference Accuracy**: User satisfaction surveys on recommendation quality
- **System Performance**: P95 < 1000ms, cache hit rate > 70%

**Success Criteria:**
- Survey completion rate > 60%
- Preference-aware search outperforms generic by 15-25% (measured via CTR)
- System handles 1000+ concurrent users
- Cost per recommendation < $0.01

---

## Appendix A — Enhanced Repo Layout
```
repo/
  infra/                 # CDK/Terraform
  services/
    main.py              # Unified FastAPI service
    models.py            # Pydantic schemas
    preference_engine.py # Preference learning algorithms
    retrieval.py         # Hybrid search implementation
    survey.py            # Survey curation logic
    cache.py             # Preference-aware caching
  data_jobs/
    embed_items.py       # Item embedding generation
    curate_survey.py     # Survey movie curation
    preference_batch.py  # Batch preference updates
  tests/
    test_preferences.py  # Preference learning tests
    test_search.py       # Search quality tests
    test_api.py          # API integration tests
  README.md
```

## Appendix B — Environment Variables
```
# Database
DDB_SWIPES_TABLE=user_swipes
DDB_PREFERENCES_TABLE=user_preferences
DDB_CACHE_TABLE=search_cache

# Retrieval
CHROMADB_PATH=/app/chroma_db
VECTOR_DIM=384
BM25_CORPUS_PATH=s3://bucket/bm25_corpus.json

# Preferences
PREFERENCE_CONFIDENCE_THRESHOLD=0.3
MIN_SWIPES_FOR_PERSONALIZATION=5
PREFERENCE_UPDATE_BATCH_SIZE=100

# Caching
CACHE_TTL_HIGH_CONFIDENCE=1800
CACHE_TTL_LOW_CONFIDENCE=900
MAX_CACHE_ENTRIES=50000

# Features
ENABLE_PREFERENCE_LEARNING=true
ENABLE_REAL_TIME_UPDATES=true
SURVEY_SIZE=20
```

This enhanced playbook transforms your recommendation system into a preference-driven engine that learns from user behavior and continuously improves recommendations through swipe-based feedback, while maintaining production-grade AWS deployment standards.