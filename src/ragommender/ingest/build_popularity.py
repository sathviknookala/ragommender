from ragommender.evaluation.build import pick_test_users, seed, test_users_n
from ragommender.catalog import read_ratings
from ragommender.paths import popularity_file
import numpy as np
import pickle

# non personal popularity prior, log(1 + rating count) scaled to 0..1 by the catalog max
# 'all' counts every user for the api, 'eval' leaves out the held out eval users so the eval can't see their ratings

def scaled(counts):
    logs = np.log1p(counts)
    return {int(m): float(v) / float(logs.max()) for m, v in logs.items()}

if __name__ == '__main__':
    ratings = read_ratings()
    _, test_users = pick_test_users(np.random.default_rng(seed), ratings)
    assert len(test_users) == test_users_n, f'expected {test_users_n} held out users, got {len(test_users)}'
    counts_all = ratings.groupby('movieId').size()
    counts_eval = ratings[~ratings['userId'].isin(test_users)].groupby('movieId').size()
    with open(popularity_file, 'wb') as f:
        pickle.dump({'all': scaled(counts_all), 'eval': scaled(counts_eval),
                     'count_eval': {int(m): int(n) for m, n in counts_eval.items()}}, f)
    print(f"{len(test_users)} held out users, {len(counts_all)} movies counted, {len(counts_eval)} after exclusion")
