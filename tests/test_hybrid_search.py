from ragommender.hybrid_search import check_same_build
from types import SimpleNamespace
import pytest

class Collection:
    def __init__(self, n):
        self.n = n

    def count(self):
        return self.n

def movieIds(n):
    return {i: f'Movie {i} (2000)' for i in range(n)}

def test_same_build_passes():
    check_same_build(Collection(3), SimpleNamespace(corpus_size=3), movieIds(3))

@pytest.mark.parametrize('collection, corpus, movies', [(3, 2, 3), (3, 3, 2), (2, 3, 3)])
def test_files_from_different_builds_fail(collection, corpus, movies):
    with pytest.raises(ValueError, match='different builds'):
        check_same_build(Collection(collection), SimpleNamespace(corpus_size=corpus), movieIds(movies))
