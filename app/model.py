import tensorflow as tf
import os
from .utils import preprocess_image

_model = None


def load_model(path="app/models/saved_model"):
    """
    Lazy-loads the TensorFlow model only once.
    Prevents high memory usage at startup.
    """
    global _model
    if _model is None:
        # Disable GPU use (Hugging Face Spaces uses only CPU)
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        print("🔹 Loading TensorFlow model from:", path)
        _model = tf.keras.models.load_model(path)
        print("✅ Model loaded successfully")
    return _model


def predict_from_image(file_storage):
    """
    Preprocesses the uploaded image and returns prediction results.
    """
    model = load_model()
    img = preprocess_image(file_storage)

    # Run inference safely
    probs = model.predict(img, verbose=0)

    # Optional: add class mapping if you have labels
    # predicted_class = CLASS_NAMES[int(tf.argmax(probs, axis=1)[0])]
    # confidence = float(tf.reduce_max(probs))
    # return {"class": predicted_class, "confidence": confidence}

    return {"raw_output": probs.tolist()}
