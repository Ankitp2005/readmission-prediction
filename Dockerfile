# Use a lightweight Python base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Hugging Face Spaces runs on port 7860 by default
EXPOSE 7860

# Command to run the application
# We point to api/app.py as the entry point
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860"]
