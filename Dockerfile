# KotoFloat Cloud Backend
# Base: Debian (NOT Alpine — MeCab C extensions fail on musl)

# --- Build stage: install heavy deps + download JMdict ---
FROM python:3.12-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements-cloud.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-cloud.txt

# Download JMdict pre-built SQLite from scriptin/jmdict-simplified
# Pin to a specific release for reproducibility
ARG JMDICT_VERSION=3.5.0
RUN mkdir -p /static && \
    curl -fSL "https://github.com/scriptin/jmdict-simplified/releases/download/${JMDICT_VERSION}/jmdict-eng-${JMDICT_VERSION}.json.tgz" \
    -o /tmp/jmdict.tgz && \
    tar -xzf /tmp/jmdict.tgz -C /tmp/ && \
    rm /tmp/jmdict.tgz
# Note: jmdict-simplified distributes JSON, not SQLite.
# We convert to SQLite during build for fast lookups.
COPY scripts/jmdict_to_sqlite.py /build/
RUN python /build/jmdict_to_sqlite.py /tmp/jmdict-eng-${JMDICT_VERSION}.json /static/jmdict.db


# --- Runtime stage ---
FROM python:3.12-slim-bookworm

# MeCab runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmecab2 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY --from=builder /static /app/static

WORKDIR /app
COPY server/ ./server/

# Persistent volume mount point for writable SQLite databases
# (session state, caches). Fly.io volume mounted here.
VOLUME /data
ENV DATA_DIR=/data
ENV JMDICT_PATH=/app/static/jmdict.db

EXPOSE 8201
CMD ["python", "-m", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8201"]
