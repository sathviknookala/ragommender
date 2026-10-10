from ragommender.paths import tmdb_file, movie_info
import asyncio
import httpx
import json
import os
import pandas as pd
import sys
import time

# fetches each movie's tmdb overview, tagline and keywords through movielens' links.csv, one movie per line
# resumable: movies already in the file are skipped, failed requests aren't written so a rerun retries them
# needs TMDB_API_KEY, a v3 api key or a v4 read access token from themoviedb.org/settings/api
links_file = movie_info / 'links.csv'
api = 'https://api.themoviedb.org/3'
# tmdb allows around 50 requests a second per ip, 429s are retried after Retry-After
concurrency = 20
retries = 6

def read_records(path=tmdb_file):
    # every record, including movies tmdb doesn't have, a line cut off by a crash is skipped and refetched
    records = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records[r['movieId']] = r
    return records

def load_overviews(path=tmdb_file):
    # movieId -> overview, only movies with a non empty one
    return {m: r['overview'] for m, r in read_records(path).items() if r['status'] == 'ok' and r['overview']}

def auth():
    key = os.environ.get('TMDB_API_KEY')
    if not key:
        sys.exit('set TMDB_API_KEY to a tmdb v3 api key or v4 read access token')
    # v4 read access tokens are jwts, v3 keys are 32 hex characters
    return ({'Authorization': f'Bearer {key}'}, {}) if key.startswith('eyJ') else ({}, {'api_key': key})

def record(movieId, tmdbId, d):
    return {'movieId': movieId, 'tmdbId': tmdbId, 'status': 'ok',
            'overview': (d.get('overview') or '').strip(), 'tagline': (d.get('tagline') or '').strip(),
            'keywords': [k['name'] for k in (d.get('keywords') or {}).get('keywords', [])],
            'title': d.get('title'), 'original_language': d.get('original_language'),
            'release_date': d.get('release_date'), 'runtime': d.get('runtime'),
            'vote_count': d.get('vote_count'), 'poster_path': d.get('poster_path')}

async def fetch(client, sem, movieId, tmdbId, params):
    async with sem:
        for attempt in range(retries):
            try:
                r = await client.get(f'{api}/movie/{tmdbId}', params=dict(params, append_to_response='keywords'))
            except httpx.TransportError:
                await asyncio.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                return record(movieId, tmdbId, r.json())
            if r.status_code == 404:
                return {'movieId': movieId, 'tmdbId': tmdbId, 'status': 'missing'}
            if r.status_code == 429 or r.status_code >= 500:
                await asyncio.sleep(float(r.headers.get('Retry-After') or 2 ** attempt))
                continue
            break
        return None

async def main(limit=None):
    headers, params = auth()
    params = dict(params, language='en-US')
    links = pd.read_csv(links_file).dropna(subset=['tmdbId'])
    done = read_records()
    todo = [(int(m), int(t)) for m, t in zip(links['movieId'], links['tmdbId']) if int(m) not in done][:limit]
    print(f"{len(links)} movies with a tmdb id, {len(done)} already fetched, {len(todo)} to go")
    async with httpx.AsyncClient(headers=headers, timeout=30, limits=httpx.Limits(max_connections=concurrency)) as client:
        # a bad key fails here once instead of once per movie
        r = await client.get(f'{api}/configuration', params=params)
        if r.status_code != 200:
            sys.exit(f'tmdb rejected the key ({r.status_code}): {r.text[:200]}')
        sem = asyncio.Semaphore(concurrency)
        start, failed = time.time(), 0
        with open(tmdb_file, 'a') as f:
            tasks = [fetch(client, sem, m, t, params) for m, t in todo]
            for i, task in enumerate(asyncio.as_completed(tasks), 1):
                rec = await task
                if rec is None:
                    failed += 1
                else:
                    f.write(json.dumps(rec) + '\n')
                if i % 1000 == 0 or i == len(todo):
                    f.flush()
                    print(f"{i}/{len(todo)} {i/(time.time()-start):.0f}/s, {failed} failed", flush=True)
    records = read_records()
    ok = [r for r in records.values() if r['status'] == 'ok']
    print(f"{len(ok)} found, {sum(bool(r['overview']) for r in ok)} with an overview, "
          f"{len(records)-len(ok)} not on tmdb, {failed} failed (rerun to retry)")

if __name__ == '__main__':
    # optional arg limits how many movies are fetched this run
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else None))
