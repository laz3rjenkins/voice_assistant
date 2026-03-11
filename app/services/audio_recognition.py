import json

import vosk
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer
from config import VOSK_MODEL_PATH, FRAME_RATE, HF_AUTH_TOKEN

from faster_whisper import WhisperModel
from config import WHISPER_MODEL_SIZE, DEVICE

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


# compute_type="float16" для GPU или "int8" для экономии памяти на CPU.
model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device=DEVICE,
    compute_type="float16" if DEVICE == "cuda" else "int8",
    use_auth_token=HF_AUTH_TOKEN,
)

def recognize_audio_by_whisper(filepath: str) -> str:
    try:
        segments, info = model.transcribe(
            filepath,
            beam_size=5,
            language="ru",
            vad_filter=True,  # Убирает тишину и шумы, повышает точность
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        result_text = ""
        for segment in segments:
            result_text += segment.text + " "

        return result_text.strip()

    except Exception as e:
        raise RuntimeError(f"Ошибка обработки аудио через Whisper: {e}")
