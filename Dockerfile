FROM nvcr.io/nvidia/cuda:12.3.0-runtime-ubuntu22.04

WORKDIR /app
RUN apt-get update && apt-get install -y python3-pip libgl1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["python", "inference.py"]
