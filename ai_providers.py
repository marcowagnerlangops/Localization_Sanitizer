from __future__ import annotations

import json
import requests


# ==========================================================
# HELPERS
# ==========================================================

def _extract_json(text: str) -> dict:
    text = (text or "").strip()

    if not text:
        raise ValueError("Empty AI response")

    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def _normalize_result(data: dict, model: str, language: str) -> dict:
    verdict = str(data.get("verdict", "REVIEW")).upper().strip()
    if verdict not in {"PASS", "REVIEW"}:
        verdict = "REVIEW"

    severity = str(data.get("severity", "Major")).strip()
    if severity not in {"OK", "Minor", "Major", "Critical"}:
        severity = "Major"

    return {
        "verdict": verdict,
        "severity": severity,
        "suggestion": str(data.get("suggestion", "")).strip(),
        "explanation": str(data.get("explanation", "")).strip(),
        "model": model,
        "language": language,
    }


# ==========================================================
# OPENAI
# ==========================================================

def review_openai(prompt, api_key, model, language):
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError(
            "Package 'openai' missing. Add openai to requirements.txt"
        ) from exc

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model=model,
        instructions=(
            "You are a localization QA reviewer. "
            "Return only JSON."
        ),
        input=prompt,
    )

    data = _extract_json(response.output_text)

    return _normalize_result(data, model, language)


# ==========================================================
# OLLAMA
# ==========================================================

def review_ollama(prompt, base_url, model, language):
    url = base_url.rstrip("/") + "/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    r = requests.post(url, json=payload, timeout=300)

    if r.status_code != 200:
        raise RuntimeError(f"Ollama error: {r.text}")

    data = r.json()
    text = data.get("response", "")

    parsed = _extract_json(text)

    return _normalize_result(parsed, model, language)


# ==========================================================
# AZURE OPENAI
# ==========================================================

def review_azure(prompt, endpoint, api_key, deployment, language):
    try:
        from openai import AzureOpenAI
    except Exception as exc:
        raise RuntimeError(
            "Package 'openai' missing. Add openai to requirements.txt"
        ) from exc

    client = AzureOpenAI(
        api_key=api_key,
        api_version="2024-02-01",
        azure_endpoint=endpoint,
    )

    response = client.responses.create(
        model=deployment,
        input=prompt,
    )

    data = _extract_json(response.output_text)

    return _normalize_result(data, deployment, language)
