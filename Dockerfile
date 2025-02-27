# Use a pre-built image with Python, Dlib, and OpenCV
FROM jhonatans01/python-dlib-opencv

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Upgrade pip before installing dependencies
RUN pip install --upgrade pip


# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose application port (if needed)
EXPOSE 5000

# Run the application
CMD ["python3", "app.py"]
