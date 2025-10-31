# ✅ Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN apt-get update && apt-get install -y libjpeg-dev zlib1g-dev \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /var/lib/apt/lists/*

# Copy all project files
COPY . .

# Ensure writable directory for Gradio (optional but safe)
RUN mkdir -p .gradio/flagged && chmod -R 777 .gradio

# Expose the Gradio default port
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]
