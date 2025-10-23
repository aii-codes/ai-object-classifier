import tensorflow as tf
from .utils import preprocess_image

_model = None


def load_model(path="app/models/saved_model"):
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(path)
    return _model


def predict_from_image(file_storage):
    model = load_model()
    img = preprocess_image(file_storage)
    probs = model.predict(img)  # shape (1, n_classes)
    # convert to readable output
    # return {'class': 'cat', 'confidence': 0.92}
    return {"raw_output": probs.tolist()}
