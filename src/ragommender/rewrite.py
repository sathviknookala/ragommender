from ragommender.catalog import genres_by_id
from ragommender import llm
from functools import cache

# the llm query rewrite, shared by the api (retrieval.search) and the eval (evaluation/run.py --natural)

@cache
def all_genres():
    return sorted({g for genres in genres_by_id().values() for g in genres} - {'(no genres listed)'})

rewrite_system = ("Rewrite a movie search into json {keywords, genres, year_from, year_to}. keywords are only "
                  "distinctive search terms: titles, themes, settings, people. Drop filler and comparative words such "
                  "as movie, film, something, like, funnier, better. year_from and year_to are only for when the movie "
                  "was released (90s movies is 1990 to 1999), and those years never go in keywords. A period the story "
                  "is set in, such as world war ii, victorian or medieval, is a keyword with null years, and so are "
                  "words like classic or old without a decade. Otherwise use null for both years. genres must come "
                  "from the allowed list.")

def rewrite_query(query: str):
    schema = {
        'type': 'object',
        'properties': {
            'keywords': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 8},
            'genres': {'type': 'array', 'items': {'type': 'string', 'enum': all_genres()}, 'maxItems': 3},
            'year_from': {'type': ['integer', 'null']},
            'year_to': {'type': ['integer', 'null']}
        },
        'required': ['keywords', 'genres', 'year_from', 'year_to']
    }
    messages = [{'role': 'system', 'content': f"{rewrite_system} Allowed genres: {', '.join(all_genres())}."},
                {'role': 'user', 'content': query}]
    # rewrites don't depend on the user so they share one cache entry per query
    return llm.cached(llm.make_key('rewrite', query.lower()),
                      lambda: llm.chat(messages, max_tokens=100, schema=schema, temperature=0.3))

# the llm side of the vocabulary mapping (vocab.py): from the tags nearest a query, keep the ones that mean what it asks
pick_system = ("Pick the tags that describe the movies a search asks for. The tags come from a fixed list movie fans "
               "wrote. Pick only tags that mean what the search asks, not ones that are merely related, broader or "
               "about something else, and pick none when none fit. Reply as json {tags: [...]}.")

def pick_tags(query: str, candidates: list):
    if not candidates:
        return [], False
    schema = {'type': 'object', 'required': ['tags'],
              'properties': {'tags': {'type': 'array', 'items': {'type': 'string', 'enum': candidates}, 'maxItems': 5}}}
    messages = [{'role': 'system', 'content': pick_system},
                {'role': 'user', 'content': f"Search: {query}\nTags: {', '.join(candidates)}"}]
    picked, was_cached = llm.cached(llm.make_key('pick', query.lower(), candidates),
                                    lambda: llm.chat(messages, max_tokens=80, schema=schema, temperature=0.0))
    return (picked['tags'] if isinstance(picked, dict) else None), was_cached
