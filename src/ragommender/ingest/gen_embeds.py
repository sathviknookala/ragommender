from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from ragommender.hybrid_search import default_embed_model
from ragommender.catalog import movies, tags
from ragommender.ingest.fetch_tmdb import load_overviews
from ragommender import paths
import chromadb
import pandas as pd
import spacy
import pickle
import re
import time
import torch
import math
import sys
start_time = time.time()

# cuda dependent 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device being used: {device}")
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")
# texts longer than this many tokens are truncated, minilm's own limit is 256
max_seq_length = 512

def create_collection(collection_name: str, movie_df: pd.DataFrame, tags_df: pd.DataFrame, k: int, embed_top_tags: int = None,
                      embed_model: str = default_embed_model, overviews: dict = None):
    '''
    Initializes chromadb collection and bm25 corpus for hybrid search.
    embed_top_tags embeds only the title, genres and that many most common distinct tags, bm25 still gets every tag
    embed_model is recorded in the collection's metadata so Retrieval encodes queries with the same model
    overviews (movieId -> tmdb overview, fetch_tmdb.load_overviews) go into the embedded text and the stored
    document after the genres, bm25 stays on title, genres and tags
    '''
    model = SentenceTransformer(embed_model, device=device)
    model.max_seq_length = min(model.max_seq_length, max_seq_length)
    print(f'{embed_model} loaded, max_seq_length {model.max_seq_length}')
    description_list = {}
    movieIds = {}
    all_texts = []
    embed_texts = []
    metadatas = []

    tags_grouped = tags_df.groupby('movieId')['tag'].apply(list).to_dict()
    top_tags = {}
    if embed_top_tags:
        counts = (tags_df.dropna(subset=['tag'])
                  .assign(tag=lambda d: d['tag'].astype(str).str.strip().str.lower())
                  .groupby(['movieId', 'tag']).size().reset_index(name='n')
                  .sort_values(['movieId', 'n', 'tag'], ascending=[True, False, True]))
        top_tags = counts.groupby('movieId')['tag'].apply(lambda t: list(t[:embed_top_tags])).to_dict()
    for movie in movie_df[:k].itertuples():
        tags = tags_grouped.get(movie.movieId, [])
        clean_tags = [str(tag) for tag in tags if not pd.isna(tag) and tag!='']
        text = f"{movie.title} {movie.genres} {' '.join(clean_tags)}"

        all_texts.append(text)
        tag_text = ' '.join(top_tags.get(movie.movieId, [])) if embed_top_tags else ' '.join(clean_tags)
        overview = (overviews or {}).get(movie.movieId)
        # without an overview this is the same text as before overviews existed
        embed_texts.append(f"{movie.title} {movie.genres} {overview + ' ' if overview else ''}{tag_text}")
        description_list[str(movie.movieId)] = text
        movieIds[movie.movieId] = movie.title

        # chroma metadata values must be scalars, so genres get one boolean flag each for where filters
        metadata = {'genres': movie.genres}
        for genre in movie.genres.split('|'):
            if genre != '(no genres listed)':
                metadata[f'genre_{genre}'] = True
        year = re.search(r'\((\d{4})\)\s*$', movie.title)
        if year:
            metadata['year'] = int(year.group(1))
        metadatas.append(metadata)

    docs = list(nlp.pipe([text.lower() for text in all_texts], batch_size=1000))
    tokens_list = [[token.text for token in doc if token.is_alpha and not token.is_stop]
                   for doc in docs]         
    bm25_index = BM25Okapi(tokens_list)

    embeddings = model.encode(
        embed_texts,    
        normalize_embeddings=True,
        batch_size=512 if embed_model == default_embed_model else 32,
        show_progress_bar=True,
        convert_to_tensor=True
        ).tolist()

    client = chromadb.PersistentClient(path=str(paths.chroma_dir))
    try:
        client.delete_collection(name=collection_name)
    except:
        pass        

    collection = client.create_collection(
        name=collection_name,
        metadata={'embed_model': embed_model, 'overviews': bool(overviews)},
        configuration={
            'hnsw': {
                'space': 'cosine',
                'max_neighbors': 16,
                'ef_construction': 200,
                'ef_search': 100,
            } 
        }
    )
    print('Creating collection')

    def batches(collection, embeddings, documents, ids, metadatas, batch_size=5400):
        for index, i in enumerate(range(0, len(embeddings), batch_size), 1):
            batch_end = min(len(embeddings), i+batch_size)

            batch_embeddings = embeddings[i:batch_end]
            batch_documents = documents[i:batch_end]
            batch_ids = ids[i:batch_end]
            batch_metadatas = metadatas[i:batch_end]

            collection.add(
                embeddings=batch_embeddings,
                documents=batch_documents,
                ids=batch_ids,
                metadatas=batch_metadatas
            )
            print(f"Added batch {index} of {math.ceil(len(embeddings)/batch_size)}")

    if collection.count() == 0:
        batches(collection, embeddings, embed_texts, list(description_list.keys()), metadatas)
    else:
        print('Collection has data')

    return collection, bm25_index, movieIds        

def overviews_arg():
    if '--overviews' not in sys.argv:
        return None
    overviews = load_overviews()
    assert overviews, f'no overviews in {paths.tmdb_file}, run ingest.fetch_tmdb first'
    print(f"{len(overviews)} of {len(movies())} movies have a tmdb overview")
    return overviews

if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    collection_name = args[0]
    # optional second arg limits the number of movies, defaults to the whole catalog
    k = int(args[1]) if len(args) > 1 else len(movies())
    # the shipped index (phase 3):
    # python -m ragommender.ingest.gen_embeds rag_db --clean-embed=25 --embed-model=Qwen/Qwen3-Embedding-0.6B
    clean = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--clean-embed=')), None)
    embed_model = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--embed-model=')), default_embed_model)
    # --overviews embeds tmdb overviews from ingest.fetch_tmdb with each movie
    overviews = overviews_arg()
    collection, bm25_index, movieIds = create_collection(collection_name, movies(), tags(), k, embed_top_tags=clean,
                                                         embed_model=embed_model, overviews=overviews)
    with open(paths.bm25_file, 'wb') as f:
        pickle.dump(bm25_index, f)
    with open(paths.movieIds_file, 'wb') as f:
        pickle.dump(movieIds, f)        

    end_time = time.time()
    print(f"Time taken: {end_time-start_time}")
