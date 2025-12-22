from flask import Flask
from routes.recognition import bp as recognition_bp

app = Flask(__name__)
app.register_blueprint(recognition_bp)

if __name__ == "__main__":
    print("Flask ready")
    app.run()
