import logging
import os

from flask import Flask
from flask_cors import CORS

from config import BASE_DIR
from routes.recognition import bp as recognition_bp

# Появится нужда хранить долго — logging.handlers.RotatingFileHandler, одна строка.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s\n%(message)s",
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "voice.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

app = Flask(__name__)
CORS(app)
app.register_blueprint(recognition_bp)

if __name__ == "__main__":
    app.run(debug=False)
