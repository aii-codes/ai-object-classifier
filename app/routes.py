from flask import Blueprint, request, jsonify, render_template
from .model import predict_from_image

bp = Blueprint("routes", __name__)


@bp.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@bp.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "no file provided"}), 400
    f = request.files["file"]
    result = predict_from_image(f)
    return jsonify(result)
