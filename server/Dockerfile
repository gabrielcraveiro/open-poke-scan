FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
# rapidocr com --no-deps: as dependências dele estão no requirements.txt, com o
# OpenCV headless no lugar do opencv-python (que pede libGL).
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir --no-deps rapidocr_onnxruntime==1.4.4
COPY main.py cardvision.py cardtext.py ./
# Gere os arquivos antes do build: scripts/export_model.py e scripts/build_index.py
# (e, opcional, scripts/build_hub_penalty.py → models/hub.json).
COPY models/ ./models/
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
