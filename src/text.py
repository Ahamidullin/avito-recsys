"""
превращаем запрос и объявление в строку для энкодера
"""
import re
from src.params import parse_item, parse_query

_spaces = re.compile(r"\s+")


def normalize(text):
    if not text:
        return ""
    text = text.lower().replace("ё", "е")
    return _spaces.sub(" ", text).strip()


def query_text(search_query, infm_params_text=None):
    return normalize(search_query)


def item_text(title, infm_params_text, description, max_services=25, max_chars=1200):
    """
    так как заголовок - самое информативное описание услуги сначала он идет 1ым, поттом вид и тип, 
    потом список услуг из прайс листа. в конце нормализую начало описания. 
    Адресс и графики работ не кладем потому что это шум

    
    """
    p = parse_item(infm_params_text)
    parts = [normalize(title)]
    if p["vid"]:
        parts.append("вид услуги: " + normalize(p["vid"]))
    if p["tip"]:
        parts.append("тип услуги: " + normalize(p["tip"]))
    if p["services"]:
        parts.append("услуги: " + ", ".join(normalize(s) for s in p["services"][:max_services]))
    text = ". ".join(parts)
    if description:
        text = text + ". " + normalize(description)
    return text[:max_chars]
