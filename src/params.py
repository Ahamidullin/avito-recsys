"""
парсер строк infm_params_text
"""
import re

# названия параметров, которые встречаются в объявлениях и в фильтрах запросов
# список собран по частотам в benchmark_items см блокнот 01
NAMES = [
    "Вид услуги", "Тип услуги", "Место оказания услуг", "Начальная цена", "Тип стоимости",
    "Опыт работы", "Стоимость", "Название услуги", "Своя услуга", "Услуга",
    "График работы, дни недели", "График работы от", "График работы до",
    "Время работы, с", "Время работы, до", "Куда выезжаете", "Работа по договору",
    "Работаете с юрлицами и ИП", "Как вы работаете", "Продолжительность",
    "Гарантия на работу", "Гарантия", "Готов закупить материалы", "Бригада",
    "Чем вы занимаетесь", "Удалённо", "Выполняю заказы от",
    "Время для связи, дни недели", "Время для связи от", "Время для связи, с",
    "Время для связи до", "Время для связи, до", "Рабочие дни", "Берёте ли срочные заказы",
    "Дни", "Признак предзаполнения прайс листа", "Кто оказывает услуги",
    "Специальность или сфера", "Специальность", "Где вы оказываете услуги", "Ваши клиенты",
    "Занятия", "Преподаватель", "Год окончания", "Минимальная сумма заказа",
    "Учебное учреждение", "Предмет или специальность", "Онлайн-показ", "Предоплата",
    "Онлайн-запись", "Рейтинг пользователя", "Доставка", "Оплата",
]

# длинные названия ставим вперед
_pattern = re.compile("(" + "|".join(re.escape(n) for n in sorted(NAMES, key=len, reverse=True)) + ")")
_names_set = set(NAMES)


def parse_params(text):
    """
    return: список пар название, значение. 
    """
    if not text:
        return []
    parts = [p.strip(" ,") for p in _pattern.split(text)]
    parts = [p for p in parts if p]
    pairs = []
    i = 0
    while i < len(parts):
        name = parts[i]
        if name not in _names_set: 
            i += 1
            continue # мусор до первого названия, пропускаем
        if i + 1 < len(parts) and parts[i + 1] not in _names_set:
            pairs.append((name, parts[i + 1]))
            i += 2
        else:
            pairs.append((name, ""))
            i += 1
    return pairs


def get_first(pairs, name):
    """1ое значение параметра с таким названием или пустая строка"""
    for n, v in pairs:
        if n == name and v:
            return v
    return ""


def get_all(pairs, name):
    """все непустые значения параметра, без повторо  в порядке появления."""
    seen = set()
    out = []
    for n, v in pairs:
        if n == name and v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def parse_item(text):
    """из параметров объявления достаем то, что нужно для поиска: вид, тип и список услуг."""
    pairs = parse_params(text)
    services = get_all(pairs, "Услуга") + get_all(pairs, "Название услуги")
    services = [s.replace("Своя услуга", "").strip() for s in services]
    services = [s for s in services if s]
    return {"vid": get_first(pairs, "Вид услуги"), "tip": get_first(pairs, "Тип услуги"), "services": services}


def parse_query(text):
    """из фильтров запроса достаем вид и тип услуги и два флага"""
    pairs = parse_params(text)
    names = [n for n, _ in pairs]
    return {"vid": get_first(pairs, "Вид услуги"), "tip": get_first(pairs, "Тип услуги"), "online": "Онлайн-запись" in names, "delivery": "Доставка" in names}
