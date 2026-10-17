# V-SyncLock artifact image (CPU only).
#   docker build -t vsynclock .
#   docker run --rm vsynclock                                   # smoke test on the shipped capture data
#   docker run --rm -v "$PWD/data:/work/data" vsynclock python scripts/run_vy_suite.py --tag final
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg

# OpenCV / MediaPipe runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python", "scripts/smoke_test.py"]
