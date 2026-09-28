"""
собираем answer.csv. 
дообученный bi энкодер + BM25 с приоритетом по локации + сверху реранкер.
эмбеддинги корпуса кэшируются в data/processed/emb. 
"""
import sys, time, pickle
sys.path.insert(0, ".")
from pathlib import Path
import numpy as np
import polars as pl
import torch
from src.ranking import Corpus, location_neighbors, query_filters
from src.rerank import item_side, candidates

MODEL = "data/models/user-bge-m3-ft1"
RERANKER = "data/models/rerank_logreg.pkl"
RAW = "data/raw"

t = time.time()
queries = pl.read_parquet(f"{RAW}/benchmark_queries.parquet")
items = pl.read_parquet(f"{RAW}/benchmark_items.parquet")
train_pairs = pl.read_parquet(f"{RAW}/train.parquet", columns=["search_location_id", "item_location_id"])
print("запросов:", len(queries), " объявлений:", len(items))

# эмбеддинги: из кэша если есть иначе считаем моделью
tag = Path(MODEL).name
cache_items = Path(f"data/processed/emb/bench_{tag}.npy")
cache_queries = Path(f"data/processed/emb/bench_queries_{tag}.npy")
if cache_items.exists() and cache_queries.exists():
    item_emb, query_emb = np.load(cache_items), np.load(cache_queries)
else:
    from sentence_transformers import SentenceTransformer
    from src.text import item_text, query_text
    from src.search import encode
    model = SentenceTransformer(MODEL, device="mps")
    model.max_seq_length = 160
    model.half()
    item_emb = encode(model, [item_text(a, b, c) for a, b, c in zip(items["item_title_raw"], items["item_infm_params_text"], items["item_description_raw"])], batch_size=64)
    query_emb = encode(model, [query_text(q) for q in queries["search_query"]], batch_size=64)
    cache_items.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_items, item_emb)
    np.save(cache_queries, query_emb)

corpus = Corpus(items, location_neighbors(train_pairs))
print("индексы построены за", round((time.time() - t) / 60, 1), "минут")

reranker = pickle.load(open(RERANKER, "rb"))
side = item_side(items)

# косинусы запрос x корпус считаем кусками, матрица целиком не нужна
q_vid = query_filters(queries)
item_t = torch.from_numpy(item_emb)
answers = []
for start in range(0, len(queries), 256):
    sims = (torch.from_numpy(query_emb[start:start + 256]) @ item_t.T).numpy()
    for k, i in enumerate(range(start, min(start + 256, len(queries)))):
        q, loc = queries["search_query"][i], queries["search_location_id"][i]
        # топ-300 по приоритету и скору.  потом реранкер выбирает из них 50
        order, feats = candidates(corpus, side, q, loc, q_vid[i], sims[k], top_n=300)
        score = reranker["model"].decision_function(reranker["scaler"].transform(feats))
        answers.append(" ".join(corpus.ids[order[j]] for j in np.argsort(-score, kind="stable")[:50]))

answer = pl.DataFrame({"query_id": queries["query_id"], "answer": answers})
answer.write_csv("answer.csv")
print("answer.csv записан. всего", round((time.time() - t) / 60, 1), "минут")
