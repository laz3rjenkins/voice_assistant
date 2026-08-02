import json
import logging

from groq import Groq

import config

client = Groq(api_key=config.GROQ_API_KEY)
logger = logging.getLogger(__name__)


def parse_text(text: str, context: str | None = None):
    context = " ".join(context.split()) if context else None

    user_content = f"Контекст: {context}\nКоманда: {text}" if context else f"Команда: {text}"

    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        # model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
        max_tokens=1024,
        stop=["<END>"], # закомментить для openai/gpt-oss-120b
    )

    output = resp.choices[0].message.content.strip() # для llama-3.3-70b-versatile
    # output = (resp.choices[0].message.content or "").split("<END>")[0].strip() # для openai/gpt-oss-120b
    output = output.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    usage = resp.usage
    logger.info(
        "ПРОМПТ\n%s\nТОКЕНЫ вход=%s выход=%s всего=%s\nОТВЕТ LLM\n%s\n",
        user_content, usage.prompt_tokens, usage.completion_tokens, usage.total_tokens, output,
    )

    try:
        command = json.loads(output)
    except json.JSONDecodeError:
        return {"kind": "none", "raw": output}

    # ponytail: проверяем только конверт. Сверку name с реестром роутов делает фронт —
    # реестр живёт там же, где router.visit(); дублировать его здесь нечем.
    if not isinstance(command, dict) or command.get("kind") not in ("nav", "form", "api", "none"):
        return {"kind": "none", "raw": output}

    return command
