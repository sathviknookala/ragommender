from concurrent.futures import ThreadPoolExecutor
import llm
import os
import pickle
import time

# rewrites eval tag queries into the kind of search a person types, to check the tag vocabulary isn't biasing the eval
natural_file = 'movie-info/eval_natural.pkl'
system = ("Turn a movie tag into a natural search a person might type into a movie app, 3 to 10 words. "
          "Keep the tag's meaning, don't add titles, names or details that aren't in the tag, and avoid copying "
          "the tag word for word when a natural phrasing differs. Reply with only the search.")

def paraphrase(tag):
    # two tries, since a busy slot returns None instead of waiting
    for _ in range(2):
        out = llm.chat([{'role': 'system', 'content': system}, {'role': 'user', 'content': tag}], max_tokens=30, temperature=0.3)
        if out:
            return out.strip().strip('"')
    return None

if __name__ == '__main__':
    with open('movie-info/eval_queries.pkl', 'rb') as f:
        queries = pickle.load(f)['queries']
    natural = {}
    if os.path.exists(natural_file):
        with open(natural_file, 'rb') as f:
            natural = pickle.load(f)
    tags = sorted({q['query'] for q in queries} - set(natural))
    start = time.time()
    with ThreadPoolExecutor(llm.max_concurrency) as pool:
        for tag, out in zip(tags, pool.map(paraphrase, tags)):
            if out:
                natural[tag] = out
    with open(natural_file, 'wb') as f:
        pickle.dump(natural, f)
    missing = len({q['query'] for q in queries} - set(natural))
    print(f"paraphrased {len(tags) - missing} tags in {time.time()-start:.0f}s, {missing} missing, {len(natural)} total")
