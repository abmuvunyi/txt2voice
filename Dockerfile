FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what's needed for inference
COPY predict.py .
COPY app.py .
COPY src/ ./src/

# Entrypoint: pass text as argument
ENTRYPOINT ["python", "predict.py"]
CMD ["Muraho, nagufasha gute uyu munsi?"]
