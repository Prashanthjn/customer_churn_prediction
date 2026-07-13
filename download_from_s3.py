import os
import boto3
from config import (
    AWS_ACCESS_KEY,
    AWS_SECRET_KEY,
    AWS_REGION,
    BUCKET_NAME,
)

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION,
)

LOCAL_DIR = "registered_model"
S3_PREFIX = "registered_model/"

os.makedirs(LOCAL_DIR, exist_ok=True)

response = s3.list_objects_v2(
    Bucket=BUCKET_NAME,
    Prefix=S3_PREFIX
)

if "Contents" in response:
    for obj in response["Contents"]:
        key = obj["Key"]

        if key.endswith("/"):
            continue

        local_path = os.path.join(
            LOCAL_DIR,
            os.path.relpath(key, S3_PREFIX)
        )

        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        print(f"Downloading {key}...")

        s3.download_file(
            BUCKET_NAME,
            key,
            local_path
        )

print("Download Complete!")