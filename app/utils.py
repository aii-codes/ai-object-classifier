from PIL import Image
import numpy as np


def preprocess_image(file_storage, target_size=(224, 224)):
    img = Image.open(file_storage).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img) / 255.0
    return arr.reshape((1,) + arr.shape)
