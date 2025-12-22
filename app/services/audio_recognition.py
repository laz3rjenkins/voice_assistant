import wave
import json
from vosk import Model, KaldiRecognizer
from config import MODEL_PATH, FRAME_CHUNK

model = Model(MODEL_PATH)

def recognize_audio(filepath: str) -> str:
    """Распознаёт текст из аудио файла и возвращает строку."""
    result_text = ""
    try:
        with wave.open(filepath, "rb") as wf:
            rec = KaldiRecognizer(model, wf.getframerate())
            while True:
                data = wf.readframes(FRAME_CHUNK)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    result_text += res.get("text", "") + " "
            final_res = json.loads(rec.FinalResult())
            result_text += final_res.get("text", "")
    except wave.Error as e:
        raise RuntimeError(f"Ошибка обработки аудио: {e}")
    return result_text.lower().strip()
