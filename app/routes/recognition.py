from flask import Blueprint, jsonify, request
import os
from config import UPLOAD_FOLDER
from services.audio_recognition import recognize_audio

bp = Blueprint("recognition", __name__)

@bp.route("/recognition/text", methods=["POST"])
def recognize_text():
    if "audio" not in request.files:
        return jsonify({"error": "Файл не найден"}), 400

    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "Имя файла пустое"}), 400

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        text = recognize_audio(filepath)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "message": f"Файл {file.filename} успешно загружен",
        # "filepath": filepath,
        "recognized_text": text
    })
