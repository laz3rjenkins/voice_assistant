import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOSK_MODEL_PATH = os.path.join(BASE_DIR, os.getenv("VOSK_MODEL_PATH", ""))
LLM_MODEL_PATH = os.path.join(BASE_DIR, os.getenv("LLM_MODEL_PATH", ""))
UPLOAD_FOLDER = os.path.join(BASE_DIR, os.getenv("UPLOAD_FOLDER", ""))
FRAME_RATE = os.getenv("FRAME_RATE", "16000")

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "")

DEVICE = "cpu" # cuda
WHISPER_MODEL_SIZE = "small" # или "medium"
HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN", "")