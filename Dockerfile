# Clinical Note Summarizer — API/UI image.
#
# Build args:
#   REQUIREMENTS=requirements.txt      (default) full runtime: torch/transformers/
#                                      bitsandbytes for serving the real checkpoint.
#   REQUIREMENTS=requirements-dev.txt  lightweight, GPU-free image for the stub
#                                      backend (no torch); used by docker-compose.
#
# The server (uvicorn) and UI (streamlit) are installed on top of either set so
# both the API and UI run from a single image regardless of the chosen base.
#
#   docker build --build-arg REQUIREMENTS=requirements-dev.txt -t clinical-notes:demo .

ARG PY_VERSION=3.11
FROM python:${PY_VERSION}-slim AS base

ARG REQUIREMENTS=requirements.txt
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# build-essential is needed by some wheels (e.g. bitsandbytes on the full build).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first for better layer caching. Always include the
# server + UI so the demo (requirements-dev) build can serve API and UI too.
COPY requirements.txt requirements-dev.txt ./
RUN pip install --upgrade pip \
    && pip install -r "${REQUIREMENTS}" \
    && pip install "uvicorn>=0.30.0" "streamlit>=1.36.0"

COPY . .

# Drop root for runtime.
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
