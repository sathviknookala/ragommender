from ragommender import catalog, paths
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import sys
import time
import torch

# the tag vocabulary search maps natural queries onto (ragommender/vocab.py): every tag at least min_users distinct
# users applied, lowercased, with its user count and an embedding from the index's model
# the api's vocabulary counts every user, --eval builds the eval's from the same tags as the eval index, without the
# held out users, so a label's own tag applications never shape the vocabulary
min_users = 3
embed_model = 'Qwen/Qwen3-Embedding-0.6B'

def build(tags_df, model):
    tags = tags_df.dropna(subset=['tag']).copy()
    tags['tag'] = tags['tag'].astype(str).str.strip().str.lower()
    users = tags.groupby('tag')['userId'].nunique()
    users = users[users >= min_users].sort_values(ascending=False)
    names = list(users.index)
    # tags are embedded as documents, without the query instruction, the way movie texts are
    emb = model.encode(names, normalize_embeddings=True, batch_size=32, show_progress_bar=True)
    return {'tags': names, 'users': users.to_numpy(), 'embeddings': np.asarray(emb, dtype=np.float32),
            'embed_model': embed_model, 'min_users': min_users}

if __name__ == '__main__':
    start = time.time()
    model = SentenceTransformer(embed_model, device='cuda' if torch.cuda.is_available() else 'cpu')
    tags_df = catalog.tags()
    out = paths.vocab_file
    if '--eval' in sys.argv:
        from ragommender.evaluation.build import pick_test_users, seed
        _, test_users = pick_test_users(np.random.default_rng(seed), catalog.read_ratings())
        tags_df = tags_df[~tags_df['userId'].isin(test_users)]
        out = paths.eval_vocab_file
    vocab = build(tags_df, model)
    with open(out, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"{len(vocab['tags'])} tags with {min_users}+ users written to {out}, {time.time()-start:.0f}s")
