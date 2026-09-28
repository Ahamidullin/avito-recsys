"""
Обучение реранкера: лог регрессия над признаками кандидатов
данные тут это валидационные запросы против корпуса валидации для каждого запроса берем топ-300 кандидатов, считаем 22 признака
на кандидата (src/rerank.py), метка - выбрал ли пользователь это объявление. обучаем на всех запросах, перед этим показываем оценку по 5 фолдам чтобы прирост был честным.
"""
import sys, time, pickle
sys.path.insert(0, ".")
from pathlib import Path
import numpy as np
import polars as pl
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sentence_transformers import SentenceTransformer
from src.text import item_text, query_text
from src.search import encode
from src.tiers import Corpus, location_neighbors, query_filters
from src.rerank import item_side, candidates, FEATURES

MODEL = "data/models/user-bge-m3-ft1"
OUT = "data/models/rerank_logreg.pkl"
TOP_N = 300
PROC = Path("data/processed")

val = pl.read_parquet(PROC / "val_queries.parquet")
corpus_df = pl.read_parquet(PROC / "val_corpus.parquet")
relevant = {q: set(r) for q, r in zip(val["query_id"], val["relevant"])}
n_rel = val["relevant"].list.len().to_numpy()
print("запросов", len(val), " объявлений", len(corpus_df))


cache = PROC / "emb" / Path(MODEL).name / "val_corpus_full.npy" # эмбеддинги корпуса валидации: из кэша или считаем
model = SentenceTransformer(MODEL, device="mps")
model.max_seq_length = 160
model.half()
if cache.exists():
    item_emb = np.load(cache)
else:
    t = time.time()
    item_emb = encode(model, [item_text(a, b, c) for a, b, c in zip(corpus_df["item_title_raw"], corpus_df["item_infm_params_text"], corpus_df["item_description_raw"])], batch_size=64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, item_emb)
    print("корпус закодирован за", round((time.time() - t) / 60), "минут")
query_emb = encode(model, [query_text(q) for q in val["search_query"]], batch_size=64)

# группы как в predict.py, соседи локаций по трейну без валидационных запросов
neighbors = location_neighbors(pl.read_parquet(PROC / "train_rest.parquet", columns=["search_location_id", "item_location_id"]))
corpus = Corpus(corpus_df, neighbors)
side = item_side(corpus_df)
q_vid = query_filters(val)

# признаки - для каждого запроса матрица 300 на 22 и метки
t = time.time()
X, y = [], []
ce = torch.from_numpy(item_emb)
qe = torch.from_numpy(query_emb)
for start in range(0, len(val), 256):
    sims = (qe[start:start + 256] @ ce.T).numpy()
    for k, i in enumerate(range(start, min(start + 256, len(val)))):
        order, feats = candidates(corpus, side, val["search_query"][i], val["search_location_id"][i], q_vid[i], sims[k], top_n=TOP_N)
        X.append(feats)
        y.append(np.array([corpus.ids[j] in relevant[val["query_id"][i]] for j in order], dtype=np.int8))
X, y = np.stack(X), np.stack(y)
print("признаки собраны за", round(time.time() - t), "с", X.shape)
print("позитив есть среди кандидатов у", round(float(y.any(axis=1).mean()), 4))


def recall50(scores, idx):
    return float(np.mean([y[q][np.argsort(-s, kind="stable")[:50]].sum() / n_rel[q] for q, s in zip(idx, scores)]))


def fit(Xtr, ytr):
    sc = StandardScaler().fit(Xtr)
    # позитивов 1 на 300, без balanced модель выучит что все негатив
    lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    return sc, lr


#учим на 4 фолдах, меряем на пятом
base, ours = [], []
rank_col = FEATURES.index("rank_fused")
for tr, te in KFold(n_splits=5, shuffle=True, random_state=0).split(np.arange(len(X))):
    sc, lr = fit(X[tr].reshape(-1, X.shape[2]), y[tr].reshape(-1))
    base.append(recall50([-X[q][:, rank_col] for q in te], te))
    ours.append(recall50([lr.decision_function(sc.transform(X[q])) for q in te], te))
print("recall@50 по 5 фолдам было", round(np.mean(base), 4), "стало", round(np.mean(ours), 4))

sc, lr = fit(X.reshape(-1, X.shape[2]), y.reshape(-1))
pickle.dump({"scaler": sc, "model": lr, "features": FEATURES}, open(OUT, "wb"))
print("модель сохранена в", OUT)
