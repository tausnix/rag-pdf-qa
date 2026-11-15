"""Why did the live app answer extractively? Probe the Gemini key directly."""
from __future__ import annotations

import requests

from src.config import settings

print("model:", settings.gemini_model)
print("key set:", bool(settings.gemini_api_key), "| len:", len(settings.gemini_api_key or ""))
url = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{settings.gemini_model}:generateContent"
)
r = requests.post(
    url,
    params={"key": settings.gemini_api_key},
    json={"contents": [{"parts": [{"text": "Reply with exactly: ok"}]}]},
    timeout=60,
)
print("status:", r.status_code)
print("body:", r.text[:900])
