import torch
import whisper

from flask import jsonify
from flask import Flask
from flask import request

from faster_whisper import WhisperModel

import speech_recognition as sr
import wave
import json
from vosk import Model, KaldiRecognizer

recognizer = sr.Recognizer()

app = Flask(__name__)
# model = whisper.load_model("turbo").to("cuda")
# model = WhisperModel("large-v3", device="cuda", compute_type="float16")


MODEL_PATH = "models/vosk-model-ru"
model = Model(MODEL_PATH)


@app.route("/recognition/text", methods=['POST'])
def recognize_text():
    if "audio" not in request.files:
        return jsonify({"error": "Файл не найден"}), 400

    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "Имя файла пустое"}), 400

    filepath = f"recieved_voices\\{file.filename}"
    file.save(filepath)

    return jsonify({
        "message": f"Файл {file.filename} успешно загружен",
        "filepath": filepath,
        "recognized_text": get_text_from_audio(filepath)
    })


def get_text_from_audio(filepath: str) -> str:
    # result = model.transcribe(filepath, fp16=True, condition_on_previous_text=False)
    #
    # return result['text'].strip()

    # THIS IS NEEDED SEGMENT
    # segments, _ = model.transcribe(filepath, language="ru", beam_size=3, temperature=0.0, without_timestamps=True)
    # text = " ".join(segment.text for segment in segments)
    # return text.strip()

    wf = wave.open(filepath, "rb")
    # if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [16000, 8000]:
    #     raise ValueError("Файл должен быть в формате WAV (Mono, 16-bit, 16kHz или 8kHz)")

    rec = KaldiRecognizer(model, wf.getframerate())
    result_text = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            result_text += res["text"] + " "
    final_res = json.loads(rec.FinalResult())
    result_text += final_res["text"]

    return result_text.strip()

    # file = sr.AudioFile(filepath)
    # with file as source:
    #     audio = recognizer.record(source)
    # recognized_text = ""
    # try:
    #     recognized_text = recognizer.recognize_google(audio, language="ru-RU")
    #     print("Text: " + recognized_text)
    # except Exception as e:
    #     print("Exception: " + str(e))
    #
    # return recognized_text


if __name__ == "__main__":
    print('flask ready')
    print(f"is available CUDA: {torch.cuda.is_available()}")
    app.run()
