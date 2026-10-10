from pathlib import Path
import os

# data lives at the repo root, not in the package, so every command works from any directory
# RAGOMMENDER_ROOT points somewhere else, e.g. a deployed copy of the data
root = Path(os.environ.get('RAGOMMENDER_ROOT', Path(__file__).resolve().parents[2]))
movie_info = root / 'movie-info'
bm25_dir = root / 'bm25'
chroma_dir = root / 'chroma'
results_dir = root / 'results'

# shipped index files, written by ingest/gen_embeds.py
bm25_file = bm25_dir / 'bm25_data.pkl'
movieIds_file = movie_info / 'movieIds.pkl'
# written by ingest/build_popularity.py
popularity_file = movie_info / 'popularity.pkl'
# written by ingest/fetch_tmdb.py
tmdb_file = movie_info / 'tmdb.jsonl'
# eval files, written by evaluation/build.py and evaluation/paraphrase.py
labels_file = movie_info / 'eval_consensus.pkl'
natural_file = movie_info / 'eval_natural.pkl'
eval_bm25_file = bm25_dir / 'eval_bm25.pkl'
eval_movieIds_file = movie_info / 'eval_movieIds.pkl'
