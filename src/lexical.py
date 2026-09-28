"""
BM25 с лемматизацией

почему он нужен рядом с dense моделью: тк он ловит точные слова

лемматизация а не стемминг тк русский стеммер режет непредсказуемо
"""
import re
from functools import lru_cache

import numpy as np
import bm25s
import pymorphy3

_morph = pymorphy3.MorphAnalyzer()
_word = re.compile(r"[а-яёa-z0-9]+")

# служебные слова которые мешают
STOP = {"и", "в", "на", "с", "по", "для", "из", "от", "до", "к", "у", "о", "не", "или", "а", "же", "the", "of", "in", "услуга", "услуги"}


@lru_cache(maxsize=1_000_000)
def lemma(word):
    return _morph.parse(word)[0].normal_form.replace("ё", "е")


def tokenize(text):
    if not text:
        return []
    words = _word.findall(text.lower())
    out = []
    for w in words:
        if len(w) < 2:
            continue
        l = lemma(w)
        if l not in STOP:
            out.append(l)
    return out


class Bm25Fields:
    """
    несколько полей с разными весами: заголовок важнее описания. для каждого поля свой индекс, итоговый скор взвешенная сумма
    """

    def __init__(self, weights):
        self.weights = weights
        self.indexes = {}

    def build(self, fields):
        for name in self.weights:
            tokens = [tokenize(t) for t in fields[name]]
            index = bm25s.BM25()
            index.index(tokens, show_progress=False)
            self.indexes[name] = index
        self.n_docs = len(fields[next(iter(self.weights))])

    def scores(self, query):
        """скор каждого документа корпуса для одного запроса"""
        q = tokenize(query)
        total = np.zeros(self.n_docs, dtype=np.float32)
        if not q:
            return total
        for name, w in self.weights.items():
            total += w * self.indexes[name].get_scores(q)
        return total

    def topk(self, queries, k=200):
        """индексы к лучших документов для каждого запроса"""
        out = np.zeros((len(queries), k), dtype=np.int64)
        for i, q in enumerate(queries):
            s = self.scores(q)
            # argpartition быстрее полной сортировки, сортируем только вверхушку
            top = np.argpartition(-s, k)[:k]
            out[i] = top[np.argsort(-s[top])]
        return out
