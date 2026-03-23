# Use official Python 3.11 slim image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (Docker caching — only reinstalls if requirements change)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
