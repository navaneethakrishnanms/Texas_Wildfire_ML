FROM python:3.11-slim

WORKDIR /app

# Build context is TDIS_REPLICA/ (this folder) -- e.g.:
#   docker build -t wildfire-api .          (run from inside TDIS_REPLICA/)
# or, when wiring up Azure Container Apps from GitHub: Dockerfile location =
# TDIS_REPLICA/Dockerfile, build context = TDIS_REPLICA.
#
# api/.env is NEVER copied into the image (see .dockerignore) -- it's
# gitignored anyway (won't exist in a cloud checkout), and config shouldn't
# be baked into an image. Real values are supplied as actual environment
# variables at container runtime:
#   - locally:  docker run --env-file api/.env -p 8000:8000 wildfire-api
#   - in Azure: Container App -> Environment variables (same key names)
# main.py reads them the same way either way via load_dotenv() + os.environ.
#
# data/ is ALSO NEVER copied into the image (see .dockerignore) -- the
# parquet files (~600MB, largest file 411MB) are gitignored and too large
# for GitHub anyway. Instead, main.py downloads them from DATA_BASE_URL
# (a public Blob Storage container) once at container startup into this
# empty /app/data directory.

COPY api/requirements.txt ./api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

COPY api/main.py ./api/main.py
RUN mkdir -p /app/data

ENV PORT=8000
EXPOSE 8000

WORKDIR /app/api
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
