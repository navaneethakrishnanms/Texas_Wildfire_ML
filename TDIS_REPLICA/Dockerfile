FROM python:3.11-slim

WORKDIR /app

# NOTE: build this from the TDIS_REPLICA/ folder (not api/) so api/ is inside
# the build context.
#
# data/ is NOT copied here on purpose: the parquet files (~600MB total,
# largest single file 411MB) are gitignored and were never pushed to GitHub
# (GitHub hard-blocks files over 100MB), so they can't exist in a cloud build
# checked out from the repo. Instead, /app/data is mounted at runtime from an
# Azure Files share (see deployment README) -- this directory just needs to
# exist so the mount has somewhere to attach.

COPY api/requirements.txt ./api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

COPY api/main.py ./api/main.py
RUN mkdir -p /app/data

ENV PORT=8000
EXPOSE 8000

WORKDIR /app/api
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
