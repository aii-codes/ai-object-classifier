import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch
import datetime
import os
import math

# --- Load model ---
tf.keras.backend.clear_session()
model = tf.keras.applications.EfficientNetB0(weights="imagenet", input_shape=(224, 224, 3))

# --- Predict function ---
def classify_image(image_path):
    # Open image from path
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_resized = image.resize((224, 224))
    arr = tf.keras.preprocessing.image.img_to_array(img_resized)
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    preds = model.predict(arr)
    decoded = tf.keras.applications.efficientnet.decode_predictions(preds, top=5)[0]
    results = {label: round(float(conf), 4) for (_, label, conf) in decoded}
    return results, image

# --- Generate PDF ---
def generate_pdf(image, results, original_path):
    # Get original filename
    base_name = os.path.splitext(os.path.basename(original_path))[0]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = f"Image_Classification_{base_name}.pdf"
    filepath = os.path.join(os.getcwd(), filename)

    pdf = canvas.Canvas(filepath, pagesize=A4)
    width, height = A4

    # --- Title (centered) ---
    title = "AI Object Classification Report"
    pdf.setFont("Helvetica-Bold", 20)
    title_width = pdf.stringWidth(title, "Helvetica-Bold", 20)
    pdf.drawString((width - title_width) / 2, height - 80, title)

    # --- Image (centered with border) ---
    img_width, img_height = 4 * inch, 4 * inch
    x_img = (width - img_width) / 2
    y_img = height - 450

    temp_path = "temp_img.jpg"
    image.save(temp_path)
    pdf.drawImage(temp_path, x_img, y_img, img_width, img_height, preserveAspectRatio=True)
    os.remove(temp_path)

    pdf.setStrokeColor(colors.black)
    pdf.rect(x_img, y_img, img_width, img_height, stroke=1, fill=0)

    # --- Predictions (smart dotted table) ---
    pdf.setFont("Helvetica", 13)
    y_table_start = y_img - 70
    table_width = 300
    x_table = (width - table_width) / 2

    line_height = 22
    table_height = len(results) * line_height + 15
    pdf.setStrokeColor(colors.black)
    pdf.rect(x_table - 15, y_table_start - table_height + 10, table_width + 30, table_height, stroke=1, fill=0)

    y_cursor = y_table_start - 10
    for label, conf in results.items():
        label_text = label.capitalize()
        value_text = f"{conf * 100:.2f}%"  # ✅ 2 decimal places

        label_w = pdf.stringWidth(label_text, "Helvetica", 13)
        value_w = pdf.stringWidth(value_text, "Helvetica", 13)
        available_space = table_width - (label_w + value_w + 20)
        dot_w = pdf.stringWidth(".", "Helvetica", 13)
        num_dots = max(0, math.floor(available_space / dot_w))
        dots = "." * num_dots

        start_x = x_table
        end_x = x_table + table_width
        pdf.drawString(start_x, y_cursor, label_text)
        pdf.drawString(start_x + label_w + 5, y_cursor, dots)
        pdf.drawRightString(end_x, y_cursor, value_text)

        y_cursor -= line_height

    # --- Footer ---
    pdf.setFont("Helvetica-Oblique", 10)
    pdf.setFillColor(colors.gray)
    pdf.drawString(70, 60, f"Generated on: {timestamp}")
    pdf.drawRightString(width - 70, 60, "Model: EfficientNetB0 (ImageNet)")
    pdf.drawCentredString(width / 2, 45, "Developed by Aii")

    pdf.save()
    return filepath

# --- Convert results to HTML ---
def results_to_html(results):
    # build custom bar-style HTML with 2 decimal places
    rows = []
    for label, conf in results.items():
        pct = conf * 100
        rows.append(f"""
        <div style="display:flex;align-items:center;margin-bottom:6px;font-family:Arial,Helvetica,sans-serif">
          <div style="width:140px">{label.capitalize()}</div>
          <div style="flex:1;background:#eee;border-radius:6px;overflow:hidden;margin:0 10px;height:18px">
            <div style="height:100%;width:{pct:.2f}%;background:linear-gradient(90deg,#4f46e5,#06b6d4);"></div>
          </div>
          <div style="width:60px;text-align:right">{pct:.2f}%</div>
        </div>
        """)
    return "<div>" + "".join(rows) + "</div>"

# --- Combined handler ---
def classify_and_download(image_path):
    if image_path is None:
        return "", gr.update(value=None, visible=False)
    results, image = classify_image(image_path)
    pdf_path = generate_pdf(image, results, image_path)

    html = results_to_html(results)
    return html, gr.update(value=pdf_path, visible=True)

# --- Clear handler ---
def clear_all():
    # return values for [html_output, download_btn]
    return "", gr.update(value=None, visible=False)

# --- Gradio Blocks ---
with gr.Blocks(title="AI Object Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "## 🧠 AI Object Classifier by Aii\nUpload an image to see the top-5 predictions from EfficientNetB0 and download a PDF report.",
    )

    with gr.Column(scale=1):
        image_input = gr.Image(type="filepath", label="Upload Image", height=300)  # Changed to filepath

        with gr.Row():
            predict_btn = gr.Button("Classify", variant="primary")
            clear_btn = gr.Button("Clear", variant="secondary")

        #label_output = gr.Label(num_top_classes=5, label="Predictions")
        html_output = gr.HTML(label="Predictions", value="")   # custom bar-style HTML
        download_btn = gr.File(label="Download Report", visible=False)

        # Connect events
        predict_btn.click(classify_and_download, inputs=image_input, outputs=[html_output, download_btn])
        clear_btn.click(clear_all, inputs=None, outputs=[html_output, download_btn])
        clear_btn.click(lambda: None, inputs=None, outputs=[image_input])  # clears input too

demo.launch(server_name="0.0.0.0", server_port=7860)
