
import numpy as np
from lsh import LSHIndex
import time

def brute_force_knn(vectors, query, k):
    d = np.linalg.norm(vectors - query, axis=1)
    idx = np.argsort(d)[:k]
    return idx, d[idx]

def main():
    n = 20000  # reduced size so it runs quickly here
    dim = 64
    vecs = np.random.randn(n, dim).astype('float32')
    ids = [f'id_{i}' for i in range(n)]
    lsh = LSHIndex(num_hash_tables=5, hash_size=10, input_dim=dim)
    lsh.index(vecs, ids)
    queries = vecs[np.random.choice(n, 5, replace=False)]
    for q in queries:
        t0 = time.time()
        res = lsh.query(q, top_k=10)
        t1 = time.time()
        bf_idx, bf_d = brute_force_knn(vecs, q, 10)
        t2 = time.time()
        print(f"LSH time: {t1-t0:.4f}s, brute time: {t2-t1:.4f}s, LSH_found={len(res)}")
        # compute recall
        lsh_ids = [r[0] for r in res]
        bf_ids = [f'id_{i}' for i in bf_idx]
        recall = len(set(lsh_ids) & set(bf_ids)) / len(bf_ids)
        print('recall:', recall)

if __name__ == '__main__':
    main()
