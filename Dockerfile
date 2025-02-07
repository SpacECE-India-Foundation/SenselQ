# Use a lightweight Python image
FROM python:3.9-slim

# Set environment variables (avoid bytecode files)
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install Git, missing libraries and other required dependencies
RUN apt update && \
    apt install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone Flask app repository
RUN git clone -b master .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask's default port
EXPOSE 5000

# Run the Flask app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
