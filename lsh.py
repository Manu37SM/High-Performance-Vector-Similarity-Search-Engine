
import numpy as np
from collections import defaultdict
import math

class LSHIndex:
    def __init__(self, num_hash_tables:int, hash_size:int, input_dim:int):
        self.num_hash_tables = num_hash_tables
        self.hash_size = hash_size
        self.input_dim = input_dim
        # hyperplanes: list of [hash_size x input_dim] arrays
        self.hyperplanes = [np.random.randn(hash_size, input_dim) for _ in range(num_hash_tables)]
        self.tables = [defaultdict(list) for _ in range(num_hash_tables)]
        self.vectors = {}  # id -> vector

    def _hash(self, vec, planes):
        # returns string key
        projections = planes.dot(vec)
        bits = (projections > 0).astype(int)
        return ''.join(bits.astype(str).tolist())

    def index(self, vectors: np.ndarray, ids):
        for i, vid in enumerate(ids):
            v = vectors[i]
            self.vectors[vid] = v
            for t, planes in enumerate(self.hyperplanes):
                key = self._hash(v, planes)
                self.tables[t][key].append(vid)

    def query(self, query_vector, top_k=5):
        candidates = set()
        for t, planes in enumerate(self.hyperplanes):
            key = self._hash(query_vector, planes)
            candidates.update(self.tables[t].get(key, []))
        if not candidates:
            return []
        # compute distances
        cand_list = list(candidates)
        dists = [(vid, np.linalg.norm(self.vectors[vid]-query_vector)) for vid in cand_list]
        dists.sort(key=lambda x: x[1])
        return dists[:top_k]
