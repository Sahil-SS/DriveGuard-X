"""Print a safe Groq connectivity diagnostic without exposing the API key."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings
from groq import Groq

print("model=" + settings.groq_model)
try:
    client = Groq(api_key=settings.groq_api_key)
    available = client.models.list()
    print("accessible_models=" + ",".join(model.id for model in available.data))
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=[{"role": "user", "content": "Return JSON with one key status and value ok."}],
        temperature=0,
        response_format={"type": "json_object"},
        timeout=20,
    )
    print("request=success")
    print(response.choices[0].message.content)
except Exception as exc:
    print("request=failed")
    print(type(exc).__name__)
    print(str(exc))
