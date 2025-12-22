import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, os.getenv("VOSK_MODEL_PATH", ""))
UPLOAD_FOLDER = os.path.join(BASE_DIR, os.getenv("UPLOAD_FOLDER", ""))
FRAME_CHUNK = os.getenv("FRAME_CHUNK", "4000")

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "")