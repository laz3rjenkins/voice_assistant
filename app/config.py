import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS_DIR = Path(BASE_DIR) / "prompts"
VOSK_MODEL_PATH = os.path.join(BASE_DIR, os.getenv("VOSK_MODEL_PATH", ""))
LLM_MODEL_PATH = os.path.join(BASE_DIR, os.getenv("LLM_MODEL_PATH", ""))
UPLOAD_FOLDER = os.path.join(BASE_DIR, os.getenv("UPLOAD_FOLDER", ""))
FRAME_RATE = os.getenv("FRAME_RATE", "16000")

# Промпт лежит в файле, а не в .env: он правится чаще всего остального, а однострочная
# переменная не даёт ни diff по строкам, ни разбора на секции для будущей сборки под фразу.
SYSTEM_PROMPT = (PROMPTS_DIR / "system.md").read_text(encoding="utf-8").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

DEVICE = "cpu" # cuda
WHISPER_MODEL_SIZE = "small" # или "medium"
HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN", "")