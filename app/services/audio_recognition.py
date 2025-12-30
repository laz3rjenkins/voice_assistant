import json

import vosk
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer
from config import VOSK_MODEL_PATH, FRAME_RATE

vosk.SetLogLevel(-1)
model = Model(VOSK_MODEL_PATH)

def recognize_audio(filepath: str) -> str:
    try:
        audio = AudioSegment.from_file(filepath)

        audio = audio.set_frame_rate(int(FRAME_RATE)).set_channels(1).set_sample_width(2)
        pcm_data = audio.raw_data

        rec = KaldiRecognizer(model, 16000)

        result_text = ""
        if rec.AcceptWaveform(pcm_data):
            res = json.loads(rec.Result())
            result_text += res.get("text", "")

        final_res = json.loads(rec.FinalResult())
        result_text += " " + final_res.get("text", "")

        return result_text.lower().strip()

    except Exception as e:
        raise RuntimeError(f"Ошибка обработки аудио: {e}")
