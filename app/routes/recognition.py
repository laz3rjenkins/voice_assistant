from flask import Blueprint, jsonify, request
import logging
import os
from datetime import datetime
from config import UPLOAD_FOLDER
from services.audio_recognition import recognize_audio_by_whisper
from services.llm_command_parser import parse_text

bp = Blueprint("recognition", __name__)
logger = logging.getLogger(__name__)

# Расширение приходит от клиента, значит в имя файла оно попадает только из белого списка:
# всё остальное уводит save() за пределы UPLOAD_FOLDER.
AUDIO_EXTENSIONS = {".webm", ".wav", ".mp4", ".m4a", ".ogg", ".mp3"}


@bp.route("/recognition/text", methods=["POST"])
def recognize_text():
    if "audio" not in request.files:
        return jsonify({"error": "Файл не найден"}), 400

    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "Имя файла пустое"}), 400

    extension = os.path.splitext(file.filename)[1].lower()
    if extension not in AUDIO_EXTENSIONS:
        extension = ".bin"

    name = datetime.now().strftime("%Y%m%d-%H%M%S-%f")

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    filepath = os.path.join(UPLOAD_FOLDER, name + extension)
    file.save(filepath)

    context = request.form.get("context") or None

    try:
        text = recognize_audio_by_whisper(filepath)
    except Exception as e:
        logger.exception("Распознавание не удалось для %s", name + extension)

        return jsonify({"error": str(e)}), 500

    with open(os.path.join(UPLOAD_FOLDER, name + ".txt"), "w", encoding="utf-8") as transcript:
        transcript.write(text)

    logger.info("АУДИО %s\nРАСШИФРОВКА\n%s", name + extension, text)

    try:
        commands = parse_text(text, context)
    except Exception as e:
        logger.exception("Разбор команды не удался для %s", name + extension)

        return jsonify({"error": str(e)}), 500

    return jsonify({
        "message": f"Файл {file.filename} успешно загружен",
        "recognized_text": text,
        "commands": commands,
    })
