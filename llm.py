from collections import OrderedDict
import hashlib
import httpx
import json
import os
import re
import threading
import time

enabled = os.environ.get('LLM_ENABLED', '1') == '1'
base_url = os.environ.get('LLM_BASE_URL', 'http://127.0.0.1:8001/v1')
model = os.environ.get('LLM_MODEL', '')
# must match the server's --max-num-seqs
max_concurrency = int(os.environ.get('LLM_MAX_CONCURRENCY', 8))
background_slots = int(os.environ.get('LLM_BACKGROUND_SLOTS', 2))
queue_timeout = float(os.environ.get('LLM_QUEUE_TIMEOUT', 2))
llm_timeout = float(os.environ.get('LLM_TIMEOUT', 20))
cooldown = float(os.environ.get('LLM_COOLDOWN', 30))
cache_size = int(os.environ.get('LLM_CACHE_SIZE', 1000))

client = httpx.Client(
    base_url=base_url,
    timeout=httpx.Timeout(llm_timeout, connect=2.0),
    limits=httpx.Limits(max_connections=max_concurrency, max_keepalive_connections=max_concurrency)
)
slots = threading.BoundedSemaphore(max_concurrency)
bg_slots = threading.BoundedSemaphore(background_slots)
model_lock = threading.Lock()
cache_lock = threading.Lock()
cache = OrderedDict()
pending = {}
down_until = 0.0
think_re = re.compile(r'<think>.*?(</think>|$)', re.DOTALL)

def get_model():
    # discover the served model once so swapping models needs no code change
    global model
    if model:
        return model
    with model_lock:
        if not model:
            resp = client.get('/models')
            resp.raise_for_status()
            model = resp.json()['data'][0]['id']
    return model

def mark_down():
    global down_until
    down_until = time.time() + cooldown

def is_up():
    return enabled and time.time() >= down_until

def chat(messages, max_tokens=200, schema=None, temperature=0.7, background=False):
    '''
    Returns the reply text, the parsed json when a schema is given, or None on any failure
    '''
    if not is_up():
        return None
    if background and not bg_slots.acquire(timeout=queue_timeout):
        return None
    try:
        if not slots.acquire(timeout=queue_timeout):
            return None
        try:
            body = {
                'model': get_model(),
                'messages': messages,
                'max_tokens': max_tokens,
                'temperature': temperature,
                'top_p': 0.8,
                'top_k': 20,
                'chat_template_kwargs': {'enable_thinking': False}
            }
            if schema:
                body['response_format'] = {'type': 'json_schema',
                                           'json_schema': {'name': 'response', 'schema': schema}}
            resp = client.post('/chat/completions', json=body)
            resp.raise_for_status()
            content = resp.json()['choices'][0]['message']['content'] or ''
            content = think_re.sub('', content).strip()
            return json.loads(content) if schema else content
        finally:
            slots.release()
    except httpx.HTTPStatusError as e:
        # 5xx means the server is unhealthy, 4xx only affects this request
        if e.response.status_code >= 500:
            mark_down()
        return None
    except httpx.TransportError:
        mark_down()
        return None
    except (json.JSONDecodeError, KeyError, IndexError):
        return None
    finally:
        if background:
            bg_slots.release()

def make_key(*parts):
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()

def cached(key, fn):
    '''
    Returns (value, was_cached). Concurrent callers with the same key wait on one generation
    '''
    with cache_lock:
        if key in cache:
            cache.move_to_end(key)
            return cache[key], True
        event = pending.get(key)
        owner = event is None
        if owner:
            event = pending[key] = threading.Event()

    if not owner:
        event.wait(llm_timeout + 2*queue_timeout)
        with cache_lock:
            return cache.get(key), key in cache

    value = None
    try:
        value = fn()
    finally:
        with cache_lock:
            # failures are not cached so the next request can retry
            if value is not None:
                cache[key] = value
                while len(cache) > cache_size:
                    cache.popitem(last=False)
            pending.pop(key, None)
            event.set()
    return value, False

def close():
    client.close()

if __name__ == '__main__':
    print(f"base_url: {base_url} up: {is_up()}")
    print(chat([{'role': 'user', 'content': 'Say hi in three words.'}], max_tokens=20))
