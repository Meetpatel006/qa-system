# Use an official Python runtime as the base image
FROM python:3.8-slim

# Set working directory in the container
WORKDIR /app

# Copy requirements first for better cache usage
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Make port 7860 available outside the container
EXPOSE 7860

# Define environment variable for Python to run in unbuffered mode
ENV PYTHONUNBUFFERED=1

# Start the Gradio application
CMD ["python", "main.py"]
