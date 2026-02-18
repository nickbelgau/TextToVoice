# Uses backblaze.com B2 storage
# Tests connectivity and list objects in a bucket.

import streamlit as st
import boto3
from botocore.config import Config


s3 = boto3.client(
    "s3",
    endpoint_url=st.secrets["B2_S3_ENDPOINT"],
    aws_access_key_id=st.secrets["B2_ACCESS_KEY_ID"],
    aws_secret_access_key=st.secrets["B2_SECRET_APPL_KEY"],
    config=Config(signature_version="s3v4"),
)

# 1) Connectivity/auth test
s3.head_bucket(Bucket=st.secrets["B2_BUCKET"])
print("✅ Connected OK")

# 2) List up to 10 objects
resp = s3.list_objects_v2(Bucket=st.secrets["B2_BUCKET"], MaxKeys=10)
for obj in resp.get("Contents", []):
    print(obj["Key"], obj["Size"])