#!/usr/bin/env python3
import os
import sys
import json
import urllib.request

DATASET_ID = "cmj8u3pir00ilnxxbgcfemllh"
OUT = "mcv-scripted-nan-tw-v24.0.tar.gz"
API = f"https://datacollective.mozillafoundation.org/api/datasets/{DATASET_ID}/download"

token = os.environ.get("MDC_API_KEY", "")
if not token:
    print("ERROR: please set env var MDC_API_KEY", file=sys.stderr)
    sys.exit(1)

# 1) POST 获取 downloadUrl
req = urllib.request.Request(
    API,
    method="POST",
    headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    },
    data=b"{}",
)

try:
    with urllib.request.urlopen(req) as r:
        payload = r.read().decode("utf-8", errors="replace")
except Exception as e:
    print("ERROR: failed to call download API:", e, file=sys.stderr)
    sys.exit(1)

try:
    obj = json.loads(payload)
except json.JSONDecodeError:
    print("ERROR: API did not return JSON:", payload[:500], file=sys.stderr)
    sys.exit(1)

url = obj.get("downloadUrl")
if not url:
    print("ERROR: downloadUrl missing. Full response:", obj, file=sys.stderr)
    sys.exit(1)

print("downloadUrl acquired.")

# 2) 下载（支持断点续传：如果已有部分文件则从末尾继续）
existing = os.path.getsize(OUT) if os.path.exists(OUT) else 0
headers = {}
if existing > 0:
    headers["Range"] = f"bytes={existing}-"
    mode = "ab"
else:
    mode = "wb"

req2 = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req2) as r, open(OUT, mode) as f:
    while True:
        chunk = r.read(1024 * 1024)
        if not chunk:
            break
        f.write(chunk)

print("Done:", OUT)
