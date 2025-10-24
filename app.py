import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load MobileNetV2 pretrained model
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Preprocess + predict function
def classify_image(image):
    image = image.resize((224, 224))
    arr = tf.keras.preprocessing.image.img_to_array(image)
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    preds = model.predict(arr)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=3)[0]

    # Format predictions as {label: confidence}
    results = {label: float(confidence) for (_, label, confidence) in decoded}
    return results

# Gradio UI
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),  # ✅ show top 3 predictions
    title="AI Object Classifier",
    description="Upload an image to see the top 3 predictions from MobileNetV2!",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
