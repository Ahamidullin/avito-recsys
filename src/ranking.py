"""
сначала объявления из города запроса, потом из соседних городов, потом все остальные.внутри каждой группы сортируем по тексту. 50 мест заполняем сверху вниз.
считать надо по всему корпусу, а не по топ-200 текстового поиска: на валидации топ-200 показывал 0.537 а вот весь корпус 0.85+
приоритет это число от 0 до 5, чем меньше тем раньше показываем:
локация: 0 своя, 1 соседняя (по трейну туда уходит >= 0.5% выборов из города запроса), 2 чужая
умножаем на 2 и прибавляем 1, если вид услуги не совпал с фильтром запроса
"""
import re
import numpy as np
import polars as pl
import bm25s

from src.params import parse_item, parse_query
from src.text import normalize
from src.lexical import Bm25Fields

NEIGHBOR_MIN_SHARE = 0.005
POPULARITY_WEIGHT = 0.1# слабая примесь числа отзывов: +0,7 пункта на валидации
TRIGRAM_WEIGHT = 0.3# символьные триграммы ловят опечатки и словоформы: +0,7 пункта


def location_neighbors(train_pairs, min_share=NEIGHBOR_MIN_SHARE):
    """
    по парам трейна.  из локации запроса А выбирали объявления из локации B в такой-то доле случаев.
    train_pairs датасет с колонками search_location_id, item_location_id
    """
    totals = train_pairs.group_by("search_location_id").agg(pl.len().alias("total"))
    trans = (train_pairs.filter(pl.col("search_location_id") != pl.col("item_location_id")).group_by("search_location_id", "item_location_id").agg(pl.len().alias("n")) .join(totals, on="search_location_id") .filter(pl.col("n") / pl.col("total") >= min_share))
    out = {}
    for a, b, n, t in trans.iter_rows():
        out.setdefault(a, []).append(b)
    return out


def trigrams(text):
    """символы триграммы по словам с границами слова"""
    t = re.sub(r"[^а-яa-z0-9 ]", " ", (text or "").lower().replace("ё", "е"))
    out = []
    for w in t.split():
        w = " " + w + " "
        out += [w[k:k + 3] for k in range(len(w) - 2)]
    return out


class Corpus:
    """индексы и признаки корпуса объявлений, считаются один раз"""
    def __init__(self, items, neighbors):
        self.ids = items["item_id"].to_list()
        parsed = [parse_item(p) for p in items["item_infm_params_text"]]
        self.loc = items["item_location_id"].to_numpy()
        self.vid = np.array([normalize(p["vid"]) for p in parsed])
        self.neighbors = neighbors
        reviews = items["item_rating_reviews_count"].fill_null(0).to_numpy()
        self.popularity = np.log1p(reviews) / np.log1p(reviews).max()

        title = items["item_title_raw"].to_list()
        services = [", ".join(p["services"]) + " " + p["vid"] + " " + p["tip"] for p in parsed]
        self.bm25 = Bm25Fields({"title": 1.0, "services": 1.0, "description": 1.0})
        self.bm25.build({"title": title, "services": services, "description": items["item_description_raw"].fill_null("").to_list()})
        self.tri = bm25s.BM25()
        self.tri.index([trigrams(a + " " + b) for a, b in zip(title, services)], show_progress=False)

    def location_priority(self, q_loc, q_vid):
        same = self.loc == q_loc
        near = np.isin(self.loc, self.neighbors.get(q_loc, [])) & ~same
        vid_ok = (self.vid == q_vid) if q_vid else np.ones(len(self.loc), dtype=bool)
        return (np.where(same, 0, np.where(near, 1, 2)) * 2 + (~vid_ok)).astype(np.int8)


def query_filters(queries):
    """вид услуги из фильтра каждого запроса, он нормализованный."""
    return [normalize(parse_query(p)["vid"]) for p in queries["search_infm_params_text"]]
