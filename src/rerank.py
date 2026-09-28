"""
признаки для переупорядочивания кандидатов внутри 50 мест. приоритет по локации и текстовый скор дают вполне хороший порядок но веса в нем подобраны рукам  а рейтин  цена,
отзывы не участвуют. здесь для каждого запроса берем топ n кандидатов текущей схемы и описываем каждого набором чисел.  дальше модель учится, что из этого предсказывает выбор.
"""
import numpy as np
import polars as pl
import torch
from src.text import normalize
from src.ranking import trigrams, TRIGRAM_WEIGHT, POPULARITY_WEIGHT

FEATURES = ["bm25", "trigram", "dense", "dense_raw", "fused", "rank_fused", "rank_bm25", "rank_dense", "tier", "same_loc", "near_loc", "vid_ok", "popularity", "rating", "has_reviews", "log_price", "price_missing", "phone_hidden", "msg_forbidden", "title_len", "desc_len", "n_services"]


def item_side(items):
    """признаки объявлений которые не зависят от запроса"""
    price = items["item_price"].cast(pl.Float64).fill_null(0).to_numpy()
    from src.params import parse_item
    n_services = np.array([len(parse_item(p)["services"]) for p in items["item_infm_params_text"]])
    return {"rating": items["item_rating"].fill_null(0).to_numpy(), "has_reviews": (items["item_rating_reviews_count"].fill_null(0).to_numpy() > 0).astype(float), "log_price": np.log1p(np.where(price > 1, price, 0)), "price_missing": (price <= 1).astype(float), "phone_hidden": items["item_is_phone_hidden"].fill_null(False).cast(pl.Int8).to_numpy().astype(float), "msg_forbidden": items["item_is_message_forbidden"].fill_null(False).cast(pl.Int8).to_numpy().astype(float), "title_len": items["item_title_raw"].fill_null("").str.len_chars().to_numpy().astype(float), "desc_len": np.log1p(items["item_description_raw"].fill_null("").str.len_chars().to_numpy().astype(float)), "n_services": n_services.astype(float)}


def candidates(corpus, side, query, q_loc, q_vid, dense_sims, top_n=300):
    """топ n по текущей схеме и матрица признаков N на len(features))"""
    q = normalize(query)
    s = corpus.bm25.scores(q); sn = s / (s.max() + 1e-9)
    tq = trigrams(q); tr = corpus.tri.get_scores(tq) if tq else np.zeros(len(s), np.float32); trn = tr / (tr.max() + 1e-9)
    dn = (dense_sims - dense_sims.min()) / (dense_sims.max() - dense_sims.min() + 1e-9)
    fused = sn + TRIGRAM_WEIGHT * trn + dn + POPULARITY_WEIGHT * corpus.popularity
    prio = corpus.location_priority(q_loc, q_vid)
    order = np.lexsort((-fused, prio))[:top_n]


    # ранги считаем среди кандидатов а не по всему корпусу тк так дешевле и достаточно
    rank_bm25 = np.argsort(np.argsort(-sn[order]))
    rank_dense = np.argsort(np.argsort(-dn[order]))
    cols = {"bm25": sn[order], "trigram": trn[order], "dense": dn[order], "dense_raw": dense_sims[order], "fused": fused[order], "rank_fused": np.arange(len(order)), "rank_bm25": rank_bm25, "rank_dense": rank_dense, "tier": prio[order].astype(float), "same_loc": (prio[order] < 2).astype(float), "near_loc": ((prio[order] >= 2) & (prio[order] < 4)).astype(float), "vid_ok": (prio[order] % 2 == 0).astype(float), "popularity": corpus.popularity[order]}
    for k, v in side.items():
        cols[k] = v[order]
    return order, np.column_stack([cols[f] for f in FEATURES]).astype(np.float32)
