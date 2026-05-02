from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional

from sanitizer_core import SegmentRecord


LANGUAGE_OPTIONS = [
    "German",
    "English",
    "French",
    "Spanish",
    "Italian",
    "Dutch",
    "Portuguese",
    "Japanese",
    "Korean",
    "Chinese Simplified",
    "Chinese Traditional",
    "Arabic",
    "Polish",
    "Czech",
    "Swedish",
    "Custom",
]

STRICTNESS_OPTIONS = ["Light", "Standard", "Strict"]
AI_REVIEW_MODES = [
    "Only Critical/Major",
    "Only Issues",
    "Only Unreviewed Issues",
    "All Segments",
]


def _extract_json(text: str) -> Dict[str, str]:
    """Parse a JSON object from model output, tolerating small wrappers."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty AI response")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def select_records_for_ai(records: List[SegmentRecord], mode: str, max_segments: int) -> List[SegmentRecord]:
    if mode == "Only Critical/Major":
        selected = [r for r in records if r.lqa_severity in {"Critical", "Major"}]
    elif mode == "Only Issues":
        selected = [r for r in records if r.issue_count > 0]
    elif mode == "Only Unreviewed Issues":
        selected = [r for r in records if r.issue_count > 0 and r.ai_status in {"", "Not reviewed"}]
    else:
        selected = list(records)

    return selected[: max(1, int(max_segments))]


def build_ai_prompt(record: SegmentRecord, target_language: str, strictness: str, custom_instructions: str = "") -> str:
    return f"""
You are a professional localization quality reviewer.
Review the target text for the target language: {target_language}.
Review strictness: {strictness}.

Check grammar, fluency, idiomatic usage, terminology risk, punctuation, and whether the target sounds natural for the requested language.
Do not rewrite protected placeholders, XML/HTML-like tags, variables, code snippets, product names, numbers, or URLs.
Only suggest a rewrite if the target has a real grammar, fluency, or idiomatic issue.
If the target is acceptable, return PASS and leave suggestion empty.

Rule-based findings already detected by the tool:
- Issue Categories: {record.issue_categories or 'None'}
- Issue Details: {record.issue_details or 'None'}
- LQA Severity: {record.lqa_severity or 'OK'}

Source language: {record.source_lang}
Target language code: {record.target_lang}
Source text:
{record.source_text}

Target text:
{record.target_text}

Additional reviewer instructions:
{custom_instructions or 'None'}

Return ONLY valid JSON with this schema:
{{
  "verdict": "PASS" or "REVIEW",
  "severity": "OK" or "Minor" or "Major" or "Critical",
  "suggestion": "rewritten target text or empty string",
  "explanation": "brief explanation, max 40 words"
}}
""".strip()


def review_segment_with_openai(
    record: SegmentRecord,
    api_key: str,
    model: str,
    target_language: str,
    strictness: str,
    custom_instructions: str = "",
) -> Dict[str, str]:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("The openai package is missing. Add 'openai' to requirements.txt.") from exc

    client = OpenAI(api_key=api_key)
    prompt = build_ai_prompt(record, target_language, strictness, custom_instructions)

    response = client.responses.create(
        model=model,
        instructions=(
            "You are a precise localization QA reviewer. "
            "Return only valid JSON. Do not include markdown."
        ),
        input=prompt,
    )

    data = _extract_json(response.output_text)

    verdict = str(data.get("verdict", "REVIEW")).strip().upper()
    if verdict not in {"PASS", "REVIEW"}:
        verdict = "REVIEW"

    severity = str(data.get("severity", "Major")).strip()
    if severity not in {"OK", "Minor", "Major", "Critical"}:
        severity = "Major"

    return {
        "verdict": verdict,
        "severity": severity,
        "suggestion": str(data.get("suggestion", "") or "").strip(),
        "explanation": str(data.get("explanation", "") or "").strip(),
        "model": model,
        "language": target_language,
    }


def apply_ai_result(record: SegmentRecord, result: Dict[str, str]) -> None:
    record.ai_status = result.get("verdict", "REVIEW")
    record.ai_severity = result.get("severity", "Major")
    record.ai_suggestion = result.get("suggestion", "")
    record.ai_explanation = result.get("explanation", "")
    record.ai_model = result.get("model", "")
    record.ai_language = result.get("language", "")
