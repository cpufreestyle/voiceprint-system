FROM python:3.11-slim

# 系统依赖
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖（利用 Docker 缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 模型缓存目录
RUN mkdir -p /app/models /app/data
ENV VP_MODEL_DIR=/app/models
ENV VP_DATA_DIR=/app/data
ENV VP_HOST=0.0.0.0
ENV VP_PORT=8700

EXPOSE 8700

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8700/api/health').raise_for_status()" || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8700"]
