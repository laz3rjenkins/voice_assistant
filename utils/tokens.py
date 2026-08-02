"""Счётчик токенов — чтобы видеть, сколько стоит промпт и каждая его секция.

    ./venv/bin/python utils/tokens.py           # разбивка SYSTEM_PROMPT по секциям
    ./venv/bin/python utils/tokens.py file.txt  # файл одним числом
    echo "текст" | ./venv/bin/python utils/tokens.py -
    ./venv/bin/python utils/tokens.py --check   # самопроверка

Точное число для конкретной модели знает только провайдер — оно приходит в
`usage` ответа Groq и пишется в voice.log на каждой команде. Здесь оценка;
её хватает, чтобы сравнивать секции между собой и прикидывать экономию.
"""

import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

try:
    # o200k_base — база токенизатора gpt-oss. Для llama/qwen расхождение
    # в единицы процентов, на сравнении секций это не сказывается.
    import tiktoken

    _encode = tiktoken.get_encoding("o200k_base").encode
except ImportError:
    _encode = None


def count(text: str) -> int:
    if _encode:
        return len(_encode(text))

    # ponytail: 3.54 символа на токен — замерено на русском SYSTEM_PROMPT
    # (13745 символов = 3871 токен по usage). `pip install tiktoken` даст точное.
    return round(len(text) / 3.54)


def by_section(prompt: str) -> list[tuple[str, int]]:
    """Разбивка по заголовкам `### `. Всё до первого заголовка — преамбула."""
    parts = re.split(r"(?m)^(### .*)$", prompt)
    rows = [("(преамбула)", count(parts[0]))]
    rows += [(head.strip("# "), count(head + body)) for head, body in zip(parts[1::2], parts[2::2])]

    return rows


def _check():
    assert count("") == 0
    assert count("привет") > 0

    rows = by_section("шапка\n### A\nтело A\n### B\nтело B")
    assert [name for name, _ in rows] == ["(преамбула)", "A", "B"], rows
    assert all(size > 0 for _, size in rows), rows

    print("ok, tiktoken" if _encode else "ok, оценка по длине (tiktoken не установлен)")


def _main(argv: list[str]):
    if argv[:1] == ["--check"]:
        return _check()

    if argv:
        text = sys.stdin.read() if argv[0] == "-" else Path(argv[0]).read_text(encoding="utf-8")
        print(count(text))

        return

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env")
    prompt = os.getenv("SYSTEM_PROMPT", "")

    if not prompt:
        sys.exit("SYSTEM_PROMPT пуст — проверьте .env")

    rows = by_section(prompt)
    total = count(prompt)

    for name, size in sorted(rows, key=lambda row: -row[1]):
        print(f"{size:6}  {100 * size / total:4.1f}%  {name}")

    print(f"{total:6}  ИТОГО{'' if _encode else ' (оценка, tiktoken не установлен)'}")


if __name__ == "__main__":
    _main(sys.argv[1:])
