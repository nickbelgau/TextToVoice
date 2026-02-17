# core/r2_store.py
import json
import boto3
from botocore.config import Config
import streamlit as st

def _cfg():
    # Works both locally (.streamlit/secrets.toml) and on Streamlit Cloud secrets UI
    return st.secrets

def client():
    c = _cfg()
    return boto3.client(
        service_name="s3",
        endpoint_url=c["R2_ENDPOINT"],
        aws_access_key_id=c["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=c["R2_SECRET_ACCESS_KEY"],
        region_name="auto",  # required by SDK, not used by R2
        config=Config(signature_version="s3v4"),
    )

def bucket():
    return _cfg()["R2_BUCKET"]

def put_bytes(key: str, data: bytes, content_type: str | None = None) -> None:
    kwargs = dict(Bucket=bucket(), Key=key, Body=data)
    if content_type:
        kwargs["ContentType"] = content_type
    client().put_object(**kwargs)

def get_bytes(key: str) -> bytes:
    r = client().get_object(Bucket=bucket(), Key=key)
    return r["Body"].read()

def exists(key: str) -> bool:
    try:
        client().head_object(Bucket=bucket(), Key=key)
        return True
    except Exception:
        return False

def put_json(key: str, obj) -> None:
    put_bytes(key, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json")

def get_json(key: str, default):
    try:
        return json.loads(get_bytes(key).decode("utf-8"))
    except Exception:
        return default

def list_keys(prefix: str) -> list[str]:
    s3 = client()
    out = []
    token = None
    while True:
        kwargs = {"Bucket": bucket(), "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for o in resp.get("Contents", []):
            out.append(o["Key"])
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return out
