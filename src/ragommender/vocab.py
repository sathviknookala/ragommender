import numpy as np
import pickle

# maps a natural query onto the index's tag vocabulary: the tags whose embeddings are nearest the query's are added to
# the bm25 text, so "uplifting movies" also searches for the words of tags like "feel-good" that the crowd wrote
# the query is embedded with tag_prompt (prompt 'tag') or reuses the knn query vector (prompt 'movie')
tag_prompt = 'Instruct: Given a movie search, retrieve tags people use for movies that match it\nQuery:'

class TagVocab:
    def __init__(self, path):
        with open(path, 'rb') as f:
            vocab = pickle.load(f)
        self.tags = vocab['tags']
        self.users = vocab['users']
        self.embeddings = vocab['embeddings']
        self.embed_model = vocab['embed_model']
        self.tag_set = set(self.tags)

    def nearest(self, query_vec, n, min_sim=0.0, min_users=1):
        # the n tags most similar to the query, among tags at least min_users people used, at min_sim cosine or more
        if n <= 0:
            return []
        sims = self.embeddings @ np.asarray(query_vec, dtype=np.float32)
        sims[self.users < min_users] = -1.0
        top = np.argpartition(-sims, n)[:n]
        top = top[np.argsort(-sims[top])]
        return [(self.tags[i], float(sims[i])) for i in top if sims[i] >= min_sim]

def map_tags(vocab, query, query_vec, params):
    # the tags a query maps to under ranking.vocab_params. a query that already is a vocabulary tag uses the index's
    # own words, and mapping it only drifts ("animation" to "cartoon"): on tag queries that cost about 0.045 on val
    if vocab is None or not params['n'] or query.strip().lower() in vocab.tag_set:
        return []
    return vocab.nearest(query_vec, int(params['n']), params['min_sim'], params['min_users'])

def expand(text, tags):
    return f"{text} {' '.join(t for t, _ in tags)}" if tags else text
