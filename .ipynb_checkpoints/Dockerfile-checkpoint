FROM python:3.9

WORKDIR /app

# ✅ Install system dependencies needed for OpenCV video (especially ffmpeg)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ✅ Install Python dependencies
RUN pip install --no-cache-dir ultralytics opencv-python

# ✅ copying the file
COPY . /app

ENTRYPOINT ["python", "run.py"]