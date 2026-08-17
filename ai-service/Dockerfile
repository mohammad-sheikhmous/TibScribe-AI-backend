FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg build-essential git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.lock .
# Experta 1.9.4 pins the source-only frozendict 1.2 package. then restore the locked
# modern setuptools before installing the rest of the environment.
RUN python -m pip install --upgrade pip \
    && python -m pip install --upgrade setuptools==78.1.0 wheel \
    && python -m pip install --no-build-isolation frozendict==1.2 schema==0.6.7 \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 \
    && python -m pip install -r requirements.lock
COPY . .
EXPOSE 8001
CMD ["sh", "-lc", "python scripts/migrate_db.py && exec uvicorn app.main:app --host 0.0.0.0 --port 8001"]
