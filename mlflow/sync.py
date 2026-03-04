#!/usr/bin/env python3
"""GCS sync helper — replaces gsutil inside the MLflow container."""
import sys
from google.cloud import storage
from google.cloud.exceptions import NotFound


def _parse(gcs_uri: str) -> tuple[str, str]:
    path = gcs_uri[5:]  # strip gs://
    bucket, _, blob = path.partition("/")
    return bucket, blob


def download(src: str, dst: str) -> bool:
    """Download src (gs://...) to dst (local path). Returns True if found."""
    bucket_name, blob_path = _parse(src)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    try:
        blob.download_to_filename(dst)
        return True
    except NotFound:
        return False


def upload(src: str, dst: str) -> None:
    """Upload src (local path) to dst (gs://...)."""
    bucket_name, blob_path = _parse(dst)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(src)


if __name__ == "__main__":
    # Usage: python3 sync.py cp <src> <dst>
    if len(sys.argv) != 4 or sys.argv[1] != "cp":
        print("Usage: sync.py cp <src> <dst>", file=sys.stderr)
        sys.exit(2)

    src, dst = sys.argv[2], sys.argv[3]
    if src.startswith("gs://"):
        found = download(src, dst)
        sys.exit(0 if found else 1)
    else:
        upload(src, dst)
