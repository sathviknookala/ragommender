from concurrent.futures import ThreadPoolExecutor
import httpx
import json
import llm
import numpy as np
import sys
import time

# realistic shapes for each llm task, roughly matching their input/output token budgets
tags = 'space alien sci-fi dark atmospheric suspense survival crew ship horror classic cult'
candidates = '\n'.join(f"{i}. Movie {i} ({1980+i}) Sci-Fi|Horror|Thriller {' '.join([tags]*6)}" for i in range(1, 6))
tasks = {
    'explain': ('You explain movie search results in one sentence each. Reply as json {item_id: reason}.',
                f"Search: scary space movies\nCandidates:\n{candidates}", 220),
    'rewrite': ('Rewrite a movie search into json {keywords: [...], genres: [...]}.',
                'something like alien but funnier, preferably from the 90s', 80),
}

def run_one(client, task, i):
    system, prompt, max_tokens = tasks[task]
    body = {
        'model': llm.get_model(),
        # vary the user turn so only the system prompt hits the prefix cache
        'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': f"[{i}] {prompt}"}],
        'max_tokens': max_tokens,
        'temperature': 0.7,
        'top_p': 0.8,
        'top_k': 20,
        'chat_template_kwargs': {'enable_thinking': False},
        'stream': True,
        'stream_options': {'include_usage': True}
    }
    start = time.time()
    first = None
    tokens = 0
    with client.stream('POST', '/chat/completions', json=body) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith('data: ') or line == 'data: [DONE]':
                continue
            chunk = json.loads(line[6:])
            if chunk.get('choices') and chunk['choices'][0]['delta'].get('content') and first is None:
                first = time.time()
            if chunk.get('usage'):
                tokens = chunk['usage']['completion_tokens']
    end = time.time()
    first = first or end
    return {'latency': end-start, 'ttft': first-start, 'tokens': tokens,
            'tok_s': (tokens-1)/(end-first) if end > first and tokens > 1 else 0.0}

def bench(task, level, rounds=2):
    client = httpx.Client(base_url=llm.base_url, timeout=120,
                          limits=httpx.Limits(max_connections=level, max_keepalive_connections=level))
    start = time.time()
    with ThreadPoolExecutor(level) as pool:
        results = list(pool.map(lambda i: run_one(client, task, i), range(level*rounds)))
    wall = time.time() - start
    client.close()

    latency = np.array([r['latency'] for r in results])
    print(f"{task:8} conc={level:<3} p50={np.percentile(latency, 50):6.2f}s p95={np.percentile(latency, 95):6.2f}s "
          f"ttft={np.mean([r['ttft'] for r in results]):5.2f}s "
          f"per_req={np.mean([r['tok_s'] for r in results]):6.1f} tok/s "
          f"total={sum(r['tokens'] for r in results)/wall:7.1f} tok/s")

if __name__ == '__main__':
    # python bench_llm.py [levels] [tasks], e.g. python bench_llm.py 1,2,4,8,12 explain,rewrite
    levels = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else '1,2,4,8,12').split(',')]
    names = (sys.argv[2] if len(sys.argv) > 2 else 'explain,rewrite').split(',')
    print(f"server: {llm.base_url} model: {llm.get_model()}")
    for task in names:
        run_one(httpx.Client(base_url=llm.base_url, timeout=120), task, -1)  # warmup
        for level in levels:
            bench(task, level)
