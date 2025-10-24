# Use a lightweight official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependencies list and install them
COPY requirements.txt .
RUN pip install --default-timeout=100 --retries=5 --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Set environment to production
ENV FLASK_ENV=production

# Expose port 5000 for Flask/Gunicorn
EXPOSE 7860

# Command to start the app using Gunicorn (production-grade server)
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:7860", "app:create_app()"]
