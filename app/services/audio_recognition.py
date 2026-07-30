from faster_whisper import WhisperModel
from config import WHISPER_MODEL_SIZE, DEVICE, HF_AUTH_TOKEN

# compute_type="float16" для GPU или "int8" для экономии памяти на CPU.
whisper_model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device=DEVICE,
    compute_type="float16" if DEVICE == "cuda" else "int8",
    use_auth_token=HF_AUTH_TOKEN,
)

def recognize_audio_by_whisper(filepath: str) -> str:
    try:
        segments, info = whisper_model.transcribe(
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
