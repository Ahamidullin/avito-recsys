"""recall@k"""


def recall_at_k(predicted, relevant, k=50):
    """
    predicted: dict query_id -> список item_id в порядке убывания скора
    relevant: dict query_id -> сет item_id
    """
    total = 0.0
    for qid, rel in relevant.items():
        top = set(predicted.get(qid, [])[:k])
        total += len(top & rel) / len(rel)
    return total / len(relevant)


def recall_by_slice(predicted, relevant, flags, k=50):
    """тот же recall, но отдельно по значениям флага (seen/unseen)"""
    out = {}
    for value in sorted(set(flags.values())):
        subset = {q: r for q, r in relevant.items() if flags[q] == value}
        out[value] = (recall_at_k(predicted, subset, k), len(subset))
    return out
