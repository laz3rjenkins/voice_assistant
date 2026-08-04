"""Сборка системного промпта под конкретную фразу.

Модели незачем видеть весь реестр: из 128 его строк к одной команде относятся единицы.
Ищем близкие строки косинусом по символьным триграммам и показываем только их —
промпт падает с ~3800 токенов до ~1100. Триграммы, а не слова, потому что Whisper
теряет буквы: «коверческое предложение», «ОО» вместо «ООО». По той же причине здесь
нет ни морфологии, ни эмбеддингов — 128 коротких строк ими не окупаются.

Самопроверка (recall по реальным расшифровкам, без обращений к Groq):

    PYTHONPATH=app ./venv/bin/python app/services/prompt_builder.py --check
"""

import math
import re
import sys
from collections import Counter
from functools import lru_cache
from typing import NamedTuple

import config

# Секции, которые нужны всегда: формат ответа, правила и запреты, разбор контекста.
ALWAYS = ("ФОРМАТ ОТВЕТА", "ОБЩИЕ ПРАВИЛА", "КОНТЕКСТ")
TOP_K = 8
TOP_EXAMPLES = 3
# Поле без операции над ним бесполезно — эти строки идут прицепом к любому найденному полю.
FIELD_OPS = ("form.set", "form.set_focused", "form.select", "form.check", "form.uncheck", "form.clear")


class Unit(NamedTuple):
    uid: str
    text: str
    """По чему ищем: обычно совпадает с text, но у полей — только подпись."""
    search: str
    group: str
    """Блок формы («Префикс requests[].details:»), чтобы не повторять его у каждого поля."""
    prefix: str = ""
    """Маршруты, на которых единица применима. Пусто — применима везде."""
    routes: tuple[str, ...] = ()


def _grams(text: str) -> Counter:
    text = " " + re.sub(r"[^a-zа-я0-9]+", " ", text.lower().replace("ё", "е")).strip() + " "

    return Counter(text[i : i + 3] for i in range(len(text) - 2))


def _cosine(query: Counter, doc: Counter, doc_norm: float) -> float:
    dot = sum(count * doc.get(gram, 0) for gram, count in query.items())

    return dot / (doc_norm or 1)


def _sections(prompt: str) -> tuple[str, dict[str, str]]:
    parts = re.split(r"(?m)^(### .*)$", prompt)

    return parts[0].strip(), {h.strip("# "): b for h, b in zip(parts[1::2], parts[2::2])}


def _unwrap(body: str) -> list[str]:
    """Склеивает перенесённые строки: новую запись начинает `имя — подпись` или «Галочки:»."""
    out: list[str] = []

    for line in body.splitlines():
        if not line.strip():
            continue

        starts = re.match(r"^\s*([a-zA-Z_][\w-]*) —", line) or line.strip().startswith("Галочки:")

        if starts or not out or re.match(r"^(Блок|Префикс)", line.strip()):
            out.append(line.rstrip())
        else:
            out[-1] += " " + line.strip()

    return out


def _registry_units(name: str, body: str, group: str) -> tuple[list[Unit], list[str]]:
    """Реестр: строка без отступа начинает команду, строки с отступом её продолжают."""
    units: list[Unit] = []
    notes: list[str] = []

    for line in body.splitlines():
        if not line.strip():
            continue

        if line.startswith(" ") and units:
            units[-1] = units[-1]._replace(
                text=units[-1].text + "\n" + line.rstrip(),
                search=units[-1].search + " " + line.strip(),
            )
        elif " — " in line:
            uid = line.split(" —")[0].strip()
            units.append(Unit(f"{group}:{uid}", line.rstrip(), line.strip(), group))
        else:
            # Пояснение ко всему реестру («Идентификатор берётся из фразы или из контекста»).
            notes.append(line.strip())

    return units, notes


def _field_units(heading: str, body: str, group: str) -> list[Unit]:
    """Поля формы. Маршруты, на которых форма открыта, написаны прямо в заголовке секции."""
    routes = tuple(re.findall(r"[\w.]+\.[\w]+", re.search(r"route=([^)]*)", heading).group(1)))
    units: list[Unit] = []
    prefix = ""

    for line in _unwrap(body):
        if re.match(r"^(Блок|Префикс)", line.strip()):
            prefix = line.strip()

            continue

        if line.strip().startswith("Галочки:"):
            for name, label in re.findall(r"([a-z_]+) \(([^)]+)\)", line):
                units.append(Unit(f"field:{name}", f"  {name} — {label}", label, group, prefix, routes))

            continue

        name, _, label = line.strip().partition(" — ")
        units.append(Unit(f"field:{name}", line.rstrip(), label or name, group, prefix, routes))

    return units


class Registry(NamedTuple):
    preamble: str
    always: str
    instruction: str
    units: list[Unit]
    norms: list[float]
    grams: list[Counter]
    notes: dict[str, list[str]]
    examples: list[tuple[str, Counter]]
    refusals: list[str]


@lru_cache(maxsize=1)
def _registry() -> Registry:
    preamble, sec = _sections(config.SYSTEM_PROMPT)
    units: list[Unit] = []
    notes: dict[str, list[str]] = {}

    for heading, body in sec.items():
        if heading.startswith("РЕЕСТР"):
            group = heading.split()[-1].lower()
            found, group_notes = _registry_units(heading, body, group)
            units += found
            notes[group] = group_notes
        elif heading.startswith("ПОЛЯ"):
            units += _field_units(heading, body, "field")

    blocks = [b.strip() for b in sec["ПРИМЕРЫ"].split("\n\n") if "Ответ:" in b]
    # Отказы держим всегда: без них модель начнёт подгонять мусор под найденные строки.
    refusals = [b for b in blocks if '"kind":"none"' in b]
    examples = [(b, _grams(b.split("Команда:")[1].split("\n")[0])) for b in blocks if b not in refusals]

    return Registry(
        preamble=preamble,
        always="\n".join(f"### {h}{sec[h]}".rstrip() for h in ALWAYS),
        instruction=f"### ИНСТРУКЦИЯ{sec['ИНСТРУКЦИЯ']}".rstrip(),
        units=units,
        norms=[math.sqrt(sum(v * v for v in _grams(u.search).values())) for u in units],
        grams=[_grams(u.search) for u in units],
        notes=notes,
        examples=examples,
        refusals=refusals,
    )


def select(phrase: str, route: str | None = None, k: int = TOP_K) -> list[Unit]:
    """Строки реестра, близкие к фразе. Маршрут отсекает поля чужой формы до поиска."""
    reg = _registry()
    query = _grams(phrase)
    scored = [
        (_cosine(query, reg.grams[i], reg.norms[i]), i)
        for i, unit in enumerate(reg.units)
        if not unit.routes or not route or route in unit.routes
    ]
    picked = [reg.units[i] for _, i in sorted(scored, key=lambda row: (-row[0], row[1]))[:k]]

    if any(unit.group == "field" for unit in picked):
        chosen = {unit.uid for unit in picked}
        picked += [u for u in reg.units if u.uid in {f"form:{op}" for op in FIELD_OPS} - chosen]

    return picked


def _render(units: list[Unit], notes: dict[str, list[str]]) -> str:
    out: list[str] = []

    for group in ("nav", "api", "form", "field"):
        chunk = [u for u in units if u.group == group]

        if not chunk:
            continue

        out += notes.get(group, [])

        if group != "field":
            out += [u.text for u in chunk]

            continue

        # Префикс блока пишем один раз на все его поля, иначе он повторяется в каждой строке.
        for prefix in dict.fromkeys(u.prefix for u in chunk):
            out.append(prefix)
            out += [u.text for u in chunk if u.prefix == prefix]

    return "\n".join(out)


def build(phrase: str, context: str | None = None, k: int = TOP_K) -> str:
    """Системный промпт под одну фразу: каркас + найденные строки реестра + примеры."""
    reg = _registry()
    route = re.search(r"route=([\w.-]+)", context or "")
    units = select(phrase, route.group(1) if route else None, k)

    query = _grams(phrase)
    examples = sorted(reg.examples, key=lambda e: -_cosine(query, e[1], 1.0))[:TOP_EXAMPLES]

    return "\n\n".join(
        [
            reg.preamble,
            reg.always,
            "### ДОСТУПНЫЕ КОМАНДЫ\n" + _render(units, reg.notes),
            "### ПРИМЕРЫ\n" + "\n\n".join([e[0] for e in examples] + reg.refusals),
            reg.instruction,
        ]
    )


# Реальные расшифровки из received_voices и строка реестра, которая обязана попасть в промпт.
CORPUS = [
    ("Открой заявку 95.", "order.index", "nav:order.edit"),
    ("Открой заказ номер 32.", "order.index", "nav:order.edit"),
    ("Привет, покажи мне ссылки на все мои заказы.", "home", "nav:order.index"),
    ("Открой вкладку данные образца.", "order.edit", "form:form.tab"),
    ("Сохрани заявку.", "order.edit", "api:order.update"),
    ("Сформируй коммерческое предложение.", "order.edit", "api:documents.offer.generate"),
    ("Сформирую коверческое предложение.", "order.edit", "api:documents.offer.generate"),
    ("Сформируй документ заявки 96.", "order.edit", "api:documents.requests.generate"),
    ("Сформирует документ заявки 94.", "order.edit", "api:documents.requests.generate"),
    ("Запиши количество образцов 5.", "order.edit", "field:amount_sample"),
    ("поставь галочку предоставить рассчитанную неопределенность.", "order.edit", "field:provide_uncertainty"),
    ("Сними галочку в испытание в области аккредитации.", "order.edit", "field:in_accreditation_scope"),
    ("Установи срок проведения испытаний на 25 августа 2027 года.", "order.edit", "field:test_deadline"),
    ("Выбери изготовителя ОО Молс сервис.", "order.edit", "field:manufacturer"),
    ("в отбор осуществляет выбери исполнитель.", "order.edit", "field:sampling_performed_by"),
    ("установить тип объекта испытаний биологические материалы.", "order.edit", "field:test_object_type"),
    ("Установи приоритет высокий.", "order.edit", "field:priority"),
    ("В наименование документа отбора запиши акт отбора.", "order.edit", "field:document_name"),
    ("количество проб три", "samples.edit", "field:probe_count"),
]


def _check():
    reg = _registry()
    assert len(reg.units) > 100, len(reg.units)
    assert reg.refusals and reg.examples

    misses = []

    for phrase, route, expected in CORPUS:
        if expected not in {u.uid for u in select(phrase, route)}:
            misses.append((phrase, expected, [u.uid for u in select(phrase, route)][:3]))

    for phrase, expected, got in misses:
        print(f"  МИМО {expected:32} {phrase[:44]!r} -> {got}")

    sizes = [len(build(p, f"route={r}")) for p, r, _ in CORPUS]
    print(f"единиц реестра {len(reg.units)}, recall@{TOP_K} {len(CORPUS) - len(misses)}/{len(CORPUS)}")
    print(f"промпт: {min(sizes)}..{max(sizes)} символов, в среднем {sum(sizes) // len(sizes)}"
          f" против {len(config.SYSTEM_PROMPT)} у полного")

    assert not misses, f"ретривер потерял {len(misses)} команд"
    print("ok")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--check"]:
        _check()
    else:
        print(build(" ".join(sys.argv[1:]) or "запиши количество образцов 5", "route=order.edit"))
