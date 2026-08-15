# One image serves all four roles: training, the API, the monitor and the dashboard. They
# share the scoring code and the config, so building them separately would mean four places
# for a dependency to drift and four chances for the monitor to compute drift against a
# different binning than the API applied. The role is chosen by the command, not the image.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# Dependencies are installed before the source is copied so that editing a module does not
# invalidate the layer that takes all the time to build.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY dashboard/ ./dashboard/
COPY config/ ./config/

# The raw data, the fitted artifact and the SQLite file are mounted at run time rather than
# baked in. Data does not belong in an image, and a model baked into a layer cannot be
# retrained without a rebuild.
RUN mkdir -p /app/data/raw /app/data/processed /app/models /app/reports

# A non-root user is the default. On a Linux host the bind mounted directories are owned by
# the host user, so if writes are refused, either run with `--user "$(id -u):$(id -g)"` or
# chown the mounted directories. On macOS and Windows the Docker file sharing layer maps
# ownership automatically and no change is needed.
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
