# VisionGuard web app: CPU-only TensorFlow serving behind gunicorn
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# 1) Dependencies FIRST: this heavy layer is cached and only rebuilt when requirements change
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

# 2) Unprivileged user to run the app
RUN useradd --create-home --uid 10001 appuser

# 3) Source code LAST: editing code only rebuilds these small layers
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser app/ app/

USER appuser
EXPOSE 8000

# Docker marks the container healthy/unhealthy using the app's own /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# Production WSGI server (not Flask's dev server); 1 worker = 1 copy of the model in RAM
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", "--timeout", "120", "app.main:create_app()"]
