# Use a lightweight official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependencies list and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Set environment to production
ENV FLASK_ENV=production

# Expose port 5000 for Flask/Gunicorn
EXPOSE 5000

# Command to start the app using Gunicorn (production-grade server)
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
