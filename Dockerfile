FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PORT=8000
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY lineguard lineguard
COPY knowledge_base knowledge_base
COPY static static
COPY scripts scripts
RUN mkdir -p data
EXPOSE 8000
# The model runs on AMD GPUs behind LLM_BASE_URL; this container is the lightweight app tier.
CMD ["sh", "-c", "uvicorn lineguard.api:app --host 0.0.0.0 --port ${PORT}"]
