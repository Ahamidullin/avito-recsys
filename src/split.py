"""
раздлелил трейн на трейн и валидацию.  берем 2500 запросов. в них входят 
запросы с текстом который остается в трейне - seen и с уникальным текстом 
"""
import polars as pl
from src.text import normalize

QUERY_COLS = ["search_query", "search_location_id", "search_is_delivery_search", "search_infm_params_text", "search_category"]


def build_validation(train, n_val=2500, seen_share=0.37, seed=42):
    """
    return (train_rest, val_queries)
    val_queries: по строке на запрос с QUERY_COLS, query_id, relevant (список item_id), is_seen
    """
    train = train.with_columns(pl.col("search_query").map_elements(normalize, return_dtype=pl.String).alias("q_norm"))
    groups = (train.group_by(QUERY_COLS).agg(pl.col("item_id").unique().alias("relevant"), pl.col("q_norm").first()))
   
    text_counts = groups.group_by("q_norm").agg(pl.len().alias("n_groups"))  # смотрим сколько разных групп у каждого текста
    groups = groups.join(text_counts, on="q_norm").sort(QUERY_COLS)

    n_seen = round(n_val * seen_share)
    n_unseen = n_val - n_seen

    # unseen - текст встречается ровно в одной группе, значит после изъятия его в трейне не останется
    unseen = groups.filter(pl.col("n_groups") == 1).sample(n=n_unseen, seed=seed) 
    # seen - текст есть еще в других группах, они остаются в трейне
    seen = groups.filter(pl.col("n_groups") >= 2).sample(n=n_seen, seed=seed)

    val = pl.concat([unseen.with_columns(pl.lit(False).alias("is_seen")),seen.with_columns(pl.lit(True).alias("is_seen")),]).drop("n_groups", "q_norm")
    val = val.with_row_index("query_id").with_columns(pl.col("query_id").cast(pl.String))

    train_rest = train.join(val.select(QUERY_COLS), on=QUERY_COLS, how="anti").drop("q_norm")
    val_items = set(it for rel in val["relevant"] for it in rel)
    train_rest = train_rest.filter(~pl.col("item_id").is_in(list(val_items)))
    return train_rest, val
