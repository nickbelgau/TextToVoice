# core/storage.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


def _clean_key(key: str) -> str:
    k = (key or "").lstrip("/").replace("\\", "/")
    if ".." in k.split("/"):
        raise ValueError("Invalid key")
    return k


class Storage:
    def write_bytes(self, key: str, data: bytes, content_type: Optional[str] = None) -> None:
        raise NotImplementedError

    def read_bytes(self, key: str) -> bytes:
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        raise NotImplementedError

    # convenience helpers
    def write_text(self, key: str, text: str, encoding: str = "utf-8") -> None:
        self.write_bytes(key, (text or "").encode(encoding), content_type="text/plain; charset=utf-8")

    def read_text(self, key: str, encoding: str = "utf-8") -> str:
        return self.read_bytes(key).decode(encoding, errors="ignore")

    def write_json(self, key: str, obj: Any) -> None:
        self.write_bytes(
            key,
            json.dumps(obj, indent=2).encode("utf-8"),
            content_type="application/json; charset=utf-8",
        )

    def read_json(self, key: str, default: Any) -> Any:
        if not self.exists(key):
            return default
        return json.loads(self.read_text(key))


@dataclass
class LocalStorage(Storage):
    root: Path  # e.g. Path.cwd() / "data"

    def _path(self, key: str) -> Path:
        return self.root / Path(_clean_key(key))

    def write_bytes(self, key: str, data: bytes, content_type: Optional[str] = None) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def read_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()


@dataclass
class B2Storage(Storage):
    endpoint_url: str
    bucket: str
    access_key_id: str
    secret_access_key: str
    prefix: str = ""  # optional, e.g. "peachy"

    def __post_init__(self) -> None:
        self.prefix = _clean_key(self.prefix).rstrip("/")
        self.s3 = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )

    def _k(self, key: str) -> str:
        key = _clean_key(key)
        return f"{self.prefix}/{key}" if self.prefix else key

    def write_bytes(self, key: str, data: bytes, content_type: Optional[str] = None) -> None:
        kwargs = {"Bucket": self.bucket, "Key": self._k(key), "Body": data}
        if content_type:
            kwargs["ContentType"] = content_type
        self.s3.put_object(**kwargs)

    def read_bytes(self, key: str) -> bytes:
        r = self.s3.get_object(Bucket=self.bucket, Key=self._k(key))
        return r["Body"].read()

    def exists(self, key: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=self._k(key))
            return True
        except ClientError as e:
            code = str(e.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise