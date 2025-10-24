import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load MobileNetV2 pretrained model
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Preprocess function
def classify_image(image):
    image = image.resize((224, 224))
    arr = tf.keras.preprocessing.image.img_to_array(image)
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    preds = model.predict(arr)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0][0]
    label = decoded[1]
    confidence = float(decoded[2])
    return {label: confidence}

# Gradio UI
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=1),
    title="AI Object Classifier",
    description="Upload an image to see what object the model predicts!",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

