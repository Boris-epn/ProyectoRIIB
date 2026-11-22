FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=5000 \
    HF_HOME=/app/.cache/huggingface \
    NLTK_DATA=/usr/local/share/nltk_data \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false \
    DENSE_BATCH_SIZE=8

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-docker.txt ./
# Instalar CPU explícitamente para no arrastrar las bibliotecas CUDA de Linux.
RUN python -m pip install --no-cache-dir torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements-docker.txt \
    && python -c "import torch; assert torch.version.cuda is None" \
    && python -m nltk.downloader -d /usr/local/share/nltk_data stopwords

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/.cache \
    && chown -R app:app /app
COPY --chown=app:app server.py dense_retrieval.py evaluation_analysis.py gunicorn.conf.py ./
COPY --chown=app:app front/ ./front/
USER app

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '5000') + '/healthz', timeout=3)"
CMD ["gunicorn", "--config", "gunicorn.conf.py", "server:app"]
