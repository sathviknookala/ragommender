# ragommender

Movie search over the MovieLens 32M catalog (87,585 movies). A query goes in and the movies that match it come out,
ranked: keyword search over the tags people gave each movie and embedding search, fused with a popularity prior.
Natural searches are mapped onto the tag vocabulary, and optional LLM features (a query rewrite and one-sentence
explanations) run on a local vLLM server. The service keeps no user state.

## How a search is ranked

1. **Vocabulary mapping.** The query is embedded with a tag lookup instruction, and the nearest tag (cosine 0.5 or
   more, among tags 100+ users applied) joins the keyword text, so "uplifting movies about friendship" also searches
   for `friendship`. A query that already is a tag is left alone.
2. **Retrieval.** BM25 (`k1 5`, `b 0.1`) over each movie's title, genres and every tag users gave it, and KNN over
   `Qwen/Qwen3-Embedding-0.6B` embeddings of the title, genres and top 25 tags (Chroma). 100 candidates from each.
3. **Fusion.** Weighted reciprocal rank fusion (BM25 1.0, vector 0.125, `k` 60) plus 0.02 × the movie's scaled
   log rating count.
4. **Optional LLM steps.** `rewrite` turns the query into keywords and a release era (a filter and a small boost) and
   lets the LLM pick among the 10 nearest tags. `explain` writes one sentence for each of the top 5. Both fall back
   to the plain ranking and template reasons when the LLM is busy or down.

All settings live in `src/ragommender/ranking.py`.

## Quality

Offline NDCG@10 on the test split of a consensus label set: a movie counts as relevant to a tag when at least 2
held out users (whose tags aren't in the eval's index) applied that tag to it. 787 tags are split by tag into val and
test, settings are picked on val, and test is scored once. Natural queries are LLM paraphrases of the same tags.

| Queries | NDCG@10 |
|---|---|
| Tags (`dark comedy`) | 0.714 |
| Natural (`dark funny movies`) | 0.403 |
| Natural, `rewrite: true` | 0.415 |

The gap between tags and natural wording is the largest one left. The rewrite's gain over the default isn't
significant (+0.0128 [−0.0018, +0.0287]) and it costs 0.7 to 1.1 s uncached, so it is opt-in. Results are in
`results/eval_consensus*.json`.

## Setup

Python 3.10 and an NVIDIA GPU. Queries can be embedded on CPU, but building the index on CPU takes hours.

```bash
pip install -e '.[dev]'       # the package, pytest and spaCy's en_core_web_sm
```

pip's torch on Linux is a CUDA build. For another CUDA version, or CPU only, install torch first following
pytorch.org.

Download [MovieLens 32M](https://files.grouplens.org/datasets/movielens/ml-32m.zip) and put `movies.csv`,
`tags.csv`, `ratings.csv` and `links.csv` in `movie-info/`. The dataset is for non-commercial use, see
`movie-info/README.txt`.

Build the index, popularity prior and tag vocabulary, in this order:

```bash
python -m ragommender.ingest.gen_embeds rag_db --clean-embed=25 --embed-model=Qwen/Qwen3-Embedding-0.6B
python -m ragommender.ingest.build_popularity
python -m ragommender.ingest.build_vocab
```

`gen_embeds` deletes and recreates the `rag_db` collection, and writes `bm25/bm25_data.pkl` and
`movie-info/movieIds.pkl` with it. The three must come from the same run, and the API refuses to start otherwise.
It takes about 7 minutes on a free 24GB GPU. It runs out of memory next to the vLLM server, so stop that first.

## Running

```bash
python -m uvicorn ragommender.api.main:app --port 8000
```

The LLM features need an OpenAI-compatible server on port 8001. `scripts/serve_llm.sh` starts vLLM with
`Qwen/Qwen3-14B-AWQ` (`VLLM_BIN` points at the `vllm` executable). Without it, search still works and the LLM
features fall back.

| Variable | Default | Meaning |
|---|---|---|
| `CHROMA_COLLECTION` | `rag_db` | The collection searched |
| `LLM_ENABLED` | `1` | `0` turns the LLM features off |
| `LLM_BASE_URL` | `http://127.0.0.1:8001/v1` | The LLM server |
| `SEARCH_LOG` | unset | A file to append one JSON line per search to |
| `DEVICE` | `cuda` when available | Where queries are embedded |
| `RAGOMMENDER_ROOT` | the repo root | Where `movie-info/`, `bm25/` and `chroma/` are |

The other LLM client settings (`LLM_TIMEOUT`, `LLM_MAX_CONCURRENCY`, ...) are at the top of `src/ragommender/llm.py`.

## API

`POST /search` takes `query` (1 to 500 characters), `k` (1 to 100, default 20), `explain` and `rewrite` (both
default `false`):

```bash
curl -s localhost:8000/search -H 'content-type: application/json' \
  -d '{"query": "uplifting movies about friendship", "k": 3}'
```

```json
{
  "items": [
    {"item_id": "318", "title": "Shawshank Redemption, The (1994)", "score": 0.03538461538461539,
     "reason": ["Keyword match for 'uplifting movies about friendship'"], "explain": null},
    {"item_id": "92259", "title": "Intouchables (2011)", ...},
    {"item_id": "2005", "title": "Goonies, The (1985)", ...}
  ],
  "model_version": "rag_db/Qwen/Qwen3-Embedding-0.6B",
  "rewritten_query": null,
  "mapped_tags": ["friendship"],
  "llm_used": false,
  "llm_cached": false
}
```

| Response field | Meaning |
|---|---|
| `items[]` | `{item_id, title, score, reason[], explain}`, best first. `item_id` is the MovieLens `movieId` |
| `reason` | Template reasons, or one LLM sentence for the top 5 with `explain` |
| `explain` | With `explain`: `{vector_rank, bm25_rank, era_boost}`, a `null` rank meaning the movie wasn't in that list |
| `mapped_tags` | Tags added to the keyword search |
| `rewritten_query` | `{keywords, genres, year_from, year_to}` when the rewrite ran |
| `llm_used`, `llm_cached` | Whether LLM output made it in, and whether all of it came from the cache |

`GET /health` answers `{status, model_version, mapping, llm}` once the index is loaded, where `llm` is `available`,
`cooldown` (a call failed lately) or `disabled`. It never calls the LLM.

## Tests

| Command | What runs |
|---|---|
| `pytest` | Unit tests: ranking math, search logic, the API contract, the LLM client, metrics. No data or GPU needed, and CI runs them on every push |
| `pytest -m integration` | The API and the eval rank identically, BM25 scoring matches `rank_bm25`, `/search` answers on `rag_db`. Needs the built indexes and the GPU with vLLM stopped |
| `pytest -m llm` | The rewrite, tag pick and explain prompts against the running LLM server |

## Offline evaluation

```bash
python -m ragommender.evaluation.build --clean-embed=25 --embed-model=Qwen/Qwen3-Embedding-0.6B   # labels and the eval index
python -m ragommender.evaluation.paraphrase       # natural wordings of the tags, needs the LLM server
python -m ragommender.ingest.build_vocab --eval   # the eval's tag vocabulary, without the held out users
python -m ragommender.evaluation.run              # tag queries
python -m ragommender.evaluation.run --natural    # natural queries and the rewrite
```

`evaluation/run.py` also sweeps settings on val (`--sweep`, `--sweep=vocab`) and scores a pick against what ships
(`--weights=vector=0.5,... --label=NAME`); its header comment lists every flag.

## Layout

```
src/ragommender/
  api/            FastAPI app (main.py) and request and response models
  retrieval.py    search, the vocabulary mapping and the LLM call sites
  hybrid_search.py, ranking.py, vocab.py, rewrite.py, llm.py
  ingest/         index, popularity, tag vocabulary and TMDB builders
  evaluation/     consensus labels, paraphrases and the eval runner
scripts/          vLLM launcher and LLM benchmark
tests/
results/          eval results
```
