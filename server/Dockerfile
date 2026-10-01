FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py cardvision.py ./
# Gere os arquivos antes do build: scripts/export_model.py e scripts/build_index.py
# (e, opcional, scripts/build_hub_penalty.py → models/hub.json).
COPY models/ ./models/
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
