"""
оценка на валидации: recall@50 с приоритетом по локации
для dense, для BM25 и для dense + BM25

.venv/bin/python eval_model.py
"""
import sys
sys.path.insert(0, ".")
from pathlib import Path
import numpy as np
import polars as pl
import torch
from sentence_transformers import SentenceTransformer
from src.text import item_text, query_text, normalize
from src.search import encode
from src.ranking import Corpus, location_neighbors, query_filters, trigrams, TRIGRAM_WEIGHT, POPULARITY_WEIGHT
from src.metrics import recall_at_k

MODEL = "data/models/user-bge-m3-ft1"
PROC = Path("data/processed")

val = pl.read_parquet(PROC / "val_queries.parquet")
corpus_df = pl.read_parquet(PROC / "val_corpus.parquet")
relevant = {q: set(r) for q, r in zip(val["query_id"], val["relevant"])}

# эмбеддинги корпуса считаются около часа поэтому решил кэшировать
model = SentenceTransformer(MODEL, device="mps")
model.max_seq_length = 160
model.half()
cache = PROC / "emb" / Path(MODEL).name / "val_corpus_full.npy"
if cache.exists():
    item_emb = np.load(cache)
else:
    item_emb = encode(model, [item_text(a, b, c) for a, b, c in zip(corpus_df["item_title_raw"], corpus_df["item_infm_params_text"], corpus_df["item_description_raw"])], batch_size=64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, item_emb)
query_emb = encode(model, [query_text(q) for q in val["search_query"]], batch_size=64)

neighbors = location_neighbors(pl.read_parquet(PROC / "train_rest.parquet", columns=["search_location_id", "item_location_id"]))
corpus = Corpus(corpus_df, neighbors)
q_vid = query_filters(val)
ce = torch.from_numpy(item_emb)
qe = torch.from_numpy(query_emb)

preds = {"dense": {}, "bm25": {}, "dense + bm25": {}}
for start in range(0, len(val), 256):
    sims = (qe[start:start + 256] @ ce.T).numpy()
    for k, i in enumerate(range(start, min(start + 256, len(val)))):
        q = normalize(val["search_query"][i])
        s = corpus.bm25.scores(q)
        bm25 = s / (s.max() + 1e-9)
        tq = trigrams(q)
        tr = corpus.tri.get_scores(tq) if tq else np.zeros(len(s), np.float32)
        tri = tr / (tr.max() + 1e-9)
        d = sims[k]
        dense = (d - d.min()) / (d.max() - d.min() + 1e-9)
        pop = POPULARITY_WEIGHT * corpus.popularity
        prio = corpus.location_priority(val["search_location_id"][i], q_vid[i])

        scores = {"dense": dense + pop, "bm25": bm25 + TRIGRAM_WEIGHT * tri + pop, "dense + bm25": bm25 + TRIGRAM_WEIGHT * tri + dense + pop}
        for name, score in scores.items():
            preds[name][val["query_id"][i]] = [corpus.ids[j] for j in np.lexsort((-score, prio))[:50]]

for name, pred in preds.items():
    print(name, round(recall_at_k(pred, relevant, 50), 4))
