FROM python:3.13-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIVO_MODEL_DIR=/models
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md LICENSE NOTICE THIRD_PARTY.md ./
COPY LICENSES ./LICENSES
COPY src ./src
RUN python -m pip install --no-cache-dir '.[server]' \
    && useradd --create-home --uid 10001 pivo
USER pivo
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["pivo", "serve", "--model-dir", "/models", "--host", "0.0.0.0", "--port", "8000"]
