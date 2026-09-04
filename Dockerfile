# ═══════════════════════════════════════════════════════════════════════════════
# Dockerfile — SLT AI Module
# ═══════════════════════════════════════════════════════════════════════════════
# Multi-stage build:
#   Stage 1 (builder)  — install all Python dependencies into a venv
#   Stage 2 (runtime)  — copy only the venv + app code into a slim final image
#
# Result: ~600 MB image vs ~1.4 GB single-stage (Prophet + pystan are heavy)
#
# Build:
#   docker build -t slt-ai-module:latest .
#
# Run standalone (development):
#   docker run -p 5000:5000 --env-file .env slt-ai-module:latest
#
# Run via docker-compose (recommended):
#   docker-compose up ai-module
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Build-time dependencies needed to compile Prophet / pystan C++ extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libgomp1 \
        git \
    && rm -rf /var/lib/apt/lists/*

# Create isolated virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy only requirements first — maximises Docker layer cache
COPY requirements.txt /tmp/requirements.txt

# Install all dependencies
# --no-cache-dir keeps the image lean
# prophet pulls in pystan which compiles C++ — can take 5–10 min on first build
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt


# ─── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Tharuka Liyanaarachchi <llcsx25311@sliit.lk>"
LABEL description="SLT After-Service AI Module — Prophet · K-Means · Dijkstra"
LABEL version="1.0.0"

# Runtime system dependencies (libgomp for Prophet's OpenMP parallelism)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
ENV VIRTUAL_ENV=/opt/venv
COPY --from=builder $VIRTUAL_ENV $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Create non-root user for security
RUN groupadd -r sltai && useradd -r -g sltai -d /app -s /sbin/nologin sltai

# Set working directory
WORKDIR /app

# Copy application source code
# .dockerignore excludes: .env, __pycache__, models/saved/*.pkl, logs/*, tests/
COPY --chown=sltai:sltai . .

# Create required runtime directories
RUN mkdir -p \
        /app/models/saved \
        /app/logs \
        /app/data/uploads \
    && chown -R sltai:sltai /app/models /app/logs /app/data/uploads

# Switch to non-root user
USER sltai

# Expose Flask port
EXPOSE 5000

# Health check — tests the /health endpoint every 30s
# Fails container if 3 consecutive checks fail (marks unhealthy in compose)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/api/ai/health || exit 1

# Environment variable defaults (override in .env or docker-compose.yml)
ENV FLASK_ENV=production \
    FLASK_DEBUG=false \
    FLASK_HOST=0.0.0.0 \
    FLASK_PORT=5000 \
    LOG_LEVEL=INFO \
    LOG_FILE=/app/logs/ai_module.log \
    MODEL_DIR=/app/models/saved

# Production entrypoint: Gunicorn with 2 sync workers
# (Prophet is CPU-bound + single-threaded; 2 workers handles concurrent requests)
# Development override in docker-compose.yml: python app.py
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "2", \
     "--worker-class", "sync", \
     "--timeout", "120", \
     "--keep-alive", "5", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info", \
     "app:app"]
