"""Web app: upload a photo in the browser, get the model's top guesses back."""
import io
import os

from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError

from common import eval_transform, load_checkpoint, predict_topk

ImageFile.LOAD_TRUNCATED_IMAGES = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "models", "skin_model.pt"))
CONFIDENCE_THRESHOLD = 0.50          # below this the page says "the model is not sure"
ALLOWED = {"png", "jpg", "jpeg", "webp", "bmp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024   # 10 MB upload limit

# Load the trained model ONCE when the server starts.
model = classes = tfm = None
if os.path.exists(MODEL_PATH):
    model, ckpt = load_checkpoint(MODEL_PATH)
    classes = ckpt["classes"]
    tfm = eval_transform(ckpt["img_size"])
    print(f"Loaded model with {len(classes)} classes from {MODEL_PATH}")
else:
    print(f"No model found at {MODEL_PATH}. Run train.py first.")


@app.route("/")
def home():
    return render_template("index.html", classes=classes or [], model_ready=model is not None)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify(error="No trained model found. Run train.py first."), 503
    f = request.files.get("file")
    if f is None or f.filename == "":
        return jsonify(error="Please choose an image first."), 400
    if f.filename.rsplit(".", 1)[-1].lower() not in ALLOWED:
        return jsonify(error="Please upload a PNG, JPG, WEBP or BMP image."), 400
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(f.read())))   # fixes sideways phone photos
        top = predict_topk(model, img, tfm, classes, k=3)                 # nothing is saved to disk
    except (UnidentifiedImageError, OSError):
        return jsonify(error="That file could not be read as an image."), 400
    return jsonify(predictions=[{"label": l, "probability": p} for l, p in top],
                   confident=top[0][1] >= CONFIDENCE_THRESHOLD)


@app.errorhandler(413)
def too_big(_):
    return jsonify(error="Image is too large (max 10 MB)."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
