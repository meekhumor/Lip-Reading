# Use a base image with necessary system dependencies
FROM python:3.9

# Install system dependencies required for dlib
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-all-dev \
    python3-dev \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip before installing Python dependencies
RUN pip install --upgrade pip

# Install dlib first to avoid conflicts
RUN pip install dlib==19.24.2 --no-cache-dir

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install remaining Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
