# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Basic build deps for numpy/scipy (smallest set that works)
RUN apt-get update && apt-get install -y build-essential gfortran && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy backend code
COPY backend /app

# Prefetch HF snapshot at build time to avoid cold-start downloads
ENV HF_REPO_ID=abir221/ctai-ctd-backend-models
RUN python -c "from huggingface_hub import snapshot_download; import os; snapshot_download(repo_id=os.environ['HF_REPO_ID'], local_dir='/app/models_cache', local_dir_use_symlinks=False)"
ENV HF_CACHE_DIR=/app/models_cache

EXPOSE 8000
CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "8000"]