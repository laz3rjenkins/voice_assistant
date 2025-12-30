from llama_cpp import Llama

import config

def parse_text(text: str):
    llm = Llama(
        model_path=config.LLM_MODEL_PATH,
        n_ctx=2048,
        n_threads=8,
        temperature=0.0,
        verbose=False,
    )

    SYSTEM_PROMPT = config.SYSTEM_PROMPT
    user_text = text
    prompt = f"""
    {SYSTEM_PROMPT}

    Команда пользователя:
    {user_text}

    Ответ:
    """

    result = llm(
        prompt,
        max_tokens=64,
        stop=["<END>"]
    )

    output = result["choices"][0]["text"].strip()

    return output
