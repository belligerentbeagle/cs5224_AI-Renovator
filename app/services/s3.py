"""S3 helpers for presigned upload/download URLs."""
from __future__ import annotations

import os

import boto3

_BUCKET = os.environ.get("S3_BUCKET", "roomstyle-cs5224")
_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")


def _client():
    ak = os.environ.get("MY_AWS_ACCESS_KEY_ID")
    sk = os.environ.get("MY_AWS_SECRET_ACCESS_KEY")
    if ak and sk:
        return boto3.client(
            "s3",
            region_name=_REGION,
            aws_access_key_id=ak,
            aws_secret_access_key=sk,
        )
    return boto3.client("s3", region_name=_REGION)


def presign_upload(s3_key: str, content_type: str, expires: int = 3600) -> str:
    """Return a presigned PUT URL so the client can upload directly to S3."""
    return _client().generate_presigned_url(
        "put_object",
        Params={"Bucket": _BUCKET, "Key": s3_key, "ContentType": content_type},
        ExpiresIn=expires,
        HttpMethod="PUT",
    )


def presign_download(s3_key: str, expires: int = 3600) -> str:
    """Return a presigned GET URL for reading a private S3 object."""
    return _client().generate_presigned_url(
        "get_object",
        Params={"Bucket": _BUCKET, "Key": s3_key},
        ExpiresIn=expires,
    )


def public_url(s3_key: str) -> str:
    """Return the public HTTPS URL for an object (bucket must allow public reads)."""
    return f"https://{_BUCKET}.s3.{_REGION}.amazonaws.com/{s3_key}"
