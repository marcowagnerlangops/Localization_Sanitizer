# sanitizer_core.py

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class SegmentRecord:
    record_id: int
    file_name: str
    file_type: str
    unit_id: str
    source_lang: str
    target_lang: str
    source_text: str
    target_text: str
    source_path: str = ""
    target_path: str = ""
    notes: str = ""
    issue_count: int = 0
    severity: str = "OK"
    issue_categories: str = ""
    issue_details: str = ""
    repair_actions: str = ""
    lqa_severity: str = "OK"
    lqa_penalty: int = 0
    lqa_details: str = ""

    # Optional AI review layer
    ai_status: str = "Not reviewed"       # Not reviewed / PASS / REVIEW / ERROR
    ai_severity: str = ""                 # OK / Minor / Major / Critical
    ai_suggestion: str = ""               # Suggested improved target text
    ai_explanation: str = ""              # Short explanation from AI
    ai_model: str = ""
    ai_language: str = ""

    meta: Dict[str, str] = field(default_factory=dict)


@dataclass
class SanitizerSettings:
    normalize_unicode: bool = True
    trim_spaces: bool = True
    remove_zero_width: bool = True
    replace_nbsp: bool = True
    collapse_spaces: bool = True
    normalize_language_codes: bool = True
    flag_tag_issues: bool = True
    flag_source_equals_target: bool = True
    flag_german_micro_qa: bool = True
    flag_brand_protection: bool = True
    flag_placeholder_issues: bool = True
    flag_number_issues: bool = True
    flag_punctuation_issues: bool = True
    flag_length_ratio: bool = True
    flag_double_ellipsis: bool = True
    flag_double_spaces: bool = True
    flag_double_dot: bool = True
    flag_space_before_period: bool = True
    flag_glossary_violations: bool = True
    enable_lqa_scoring: bool = True


LANGUAGE_CODE_MAP = {
    "de": "de-DE", "de-de": "de-DE", "de_de": "de-DE", "deu": "de-DE", "ger": "de-DE",
    "en": "en-US", "en-us": "en-US", "en_us": "en-US", "eng": "en-US",
    "en-gb": "en-GB", "en_gb": "en-GB", "en-uk": "en-GB",
    "fr": "fr-FR", "fr-ca": "fr-CA", "es": "es-ES", "es-mx": "es-MX",
    "pt": "pt-PT", "pt-br": "pt-BR", "it": "it-IT", "nl": "nl-NL",
    "ja": "ja-JP", "ko": "ko-KR", "zh": "zh-CN", "zh-tw": "zh-TW",
}


def normalize_language_code(code: str) -> str:
    raw = (code or "").strip()
    if not raw:
        return ""
    key = raw.lower().replace("_", "-")
    return LANGUAGE_CODE_MAP.get(key, raw.replace("_", "-"))


def is_german(code: str) -> bool:
    return (code or "").lower().startswith("de")


class BrandRules:
    def __init__(self):
        self.rules = []

    def load_from_dataframe(self, df):
        if df.shape[1] < 2:
            raise ValueError("Brand file needs Column A = source term and Column B = required target representation.")
        self.rules = []
        for _, row in df.iterrows():
            source = "" if row.iloc[0] is None else str(row.iloc[0]).strip()
            target = "" if row.iloc[1] is None else str(row.iloc[1]).strip()
            if source and target and source.lower() not in {"source", "source term"}:
                self.rules.append({"source": source, "required": target})
        return len(self.rules)


class GlossaryRules:
    def __init__(self):
        self.rules = []

    def load_from_dataframe(self, df):
        if df.shape[1] < 2:
            raise ValueError("Glossary needs Column A = source term and Column B = required target term.")
        self.rules = []
        for _, row in df.iterrows():
            source = "" if row.iloc[0] is None else str(row.iloc[0]).strip()
            target = "" if row.iloc[1] is None else str(row.iloc[1]).strip()
            if source and target and source.lower() not in {"source", "source term"}:
                self.rules.append({"source": source, "required": target})
        return len(self.rules)


class RepairEngine:
    ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff]")

    @staticmethod
    def repair_text(text, settings):
        value = text or ""
        actions = []
        before = value
        if settings.normalize_unicode:
            value = unicodedata.normalize("NFC", value)
            if value != before:
                actions.append("Unicode normalized")
        before = value
        if settings.replace_nbsp:
            value = value.replace("\xa0", " ")
            if value != before:
                actions.append("NBSP replaced")
        before = value
        if settings.remove_zero_width:
            value = RepairEngine.ZERO_WIDTH.sub("", value)
            if value != before:
                actions.append("Zero-width removed")
        before = value
        if settings.collapse_spaces:
            value = re.sub(r"[ \t]{2,}", " ", value)
            if value != before:
                actions.append("Repeated spaces collapsed")
        before = value
        if settings.trim_spaces:
            value = value.strip()
            if value != before:
                actions.append("Trimmed")
        return value, actions

    @staticmethod
    def apply(records, settings):
        changed = 0
        for r in records:
            before = (r.source_text, r.target_text, r.source_lang, r.target_lang)
            src, a1 = RepairEngine.repair_text(r.source_text, settings)
            tgt, a2 = RepairEngine.repair_text(r.target_text, settings)
            r.source_text = src
            r.target_text = tgt
            actions = [f"Source: {x}" for x in a1] + [f"Target: {x}" for x in a2]
            if settings.normalize_language_codes:
                old1 = r.source_lang
                old2 = r.target_lang
                r.source_lang = normalize_language_code(r.source_lang)
                r.target_lang = normalize_language_code(r.target_lang)
                if old1 != r.source_lang:
                    actions.append(f"Source lang {old1}->{r.source_lang}")
                if old2 != r.target_lang:
                    actions.append(f"Target lang {old2}->{r.target_lang}")
            r.repair_actions = "; ".join(actions)
            after = (r.source_text, r.target_text, r.source_lang, r.target_lang)
            if before != after:
                changed += 1
        return changed


LQA_WEIGHTS = {"Critical": 10, "Major": 5, "Minor": 1, "OK": 0}
LQA_RANK = {"OK": 0, "Minor": 1, "Major": 2, "Critical": 3}


def worst_lqa_severity(severities: List[str]) -> str:
    if not severities:
        return "OK"
    return max(severities, key=lambda s: LQA_RANK.get(s, 0))


def quality_label(score: int) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 80:
        return "Pass"
    if score >= 70:
        return "Review Required"
    return "Fail"


def calculate_quality_score(records: List[SegmentRecord]) -> Tuple[int, int]:
    total_penalty = sum(max(0, int(r.lqa_penalty or 0)) for r in records)
    score = max(0, 100 - total_penalty)
    return score, total_penalty


class QAEngine:
    PLACEHOLDER_PATTERNS = [r"\{\d+\}", r"\{[A-Za-z0-9_]+\}", r"%s", r"%d", r"<[^>]+>"]
    TAG_PATTERN = re.compile(r"</?([A-Za-z][A-Za-z0-9:_-]*)(?:\s[^>]*)?>")

    @staticmethod
    def placeholders(text):
        found = []
        for patt in QAEngine.PLACEHOLDER_PATTERNS:
            found.extend(re.findall(patt, text or ""))
        return sorted(found)

    @staticmethod
    def numbers(text):
        return re.findall(r"\d+(?:[\.,]\d+)?", text or "")

    @staticmethod
    def end_punct(text):
        text = (text or "").strip()
        return text[-1] if text and text[-1] in ".,:;!?" else ""

    @staticmethod
    def has_tag_issue(text):
        text = text or ""
        if "<" not in text and ">" not in text:
            return False
        if text.count("<") != text.count(">"):
            return True
        stack = []
        for m in QAEngine.TAG_PATTERN.finditer(text):
            full = m.group(0)
            tag = m.group(1).lower()
            if full.endswith("/>") or tag in {"br", "hr", "img", "input", "meta", "link"}:
                continue
            if full.startswith("</"):
                if not stack or stack[-1] != tag:
                    return True
                stack.pop()
            else:
                stack.append(tag)
        return bool(stack)

    @staticmethod
    def add_issue(issues, cats, lqa_items, category, message, severity):
        issues.append(message)
        cats.append(category)
        lqa_items.append((severity, message))

    @staticmethod
    def brand_violations(source, target, brand_rules):
        issues = []
        for rule in brand_rules.rules:
            patt = r"\b" + re.escape(rule["source"]) + r"\b"
            if re.search(patt, source or "", flags=re.IGNORECASE):
                required = r"\b" + re.escape(rule["required"]) + r"\b"
                if not re.search(required, target or "", flags=re.IGNORECASE):
                    issues.append(f"Protected term '{rule['source']}' should be '{rule['required']}'")
        return issues

    @staticmethod
    def glossary_violations(source, target, glossary_rules):
        issues = []
        for rule in glossary_rules.rules:
            source_patt = r"\b" + re.escape(rule["source"]) + r"\b"
            if re.search(source_patt, source or "", flags=re.IGNORECASE):
                target_patt = r"\b" + re.escape(rule["required"]) + r"\b"
                if not re.search(target_patt, target or "", flags=re.IGNORECASE):
                    issues.append(f"Glossary violation: {rule['source']} -> {rule['required']}")
        return issues

    @staticmethod
    def german_micro_issues(record):
        if not is_german(record.target_lang):
            return []
        target = record.target_text or ""
        issues = []
        if re.search(r"\s+[.,:;!?]", target):
            issues.append("German QA: space before punctuation")
        if re.search(r"\b(\w+)\s+\1\b", target, flags=re.IGNORECASE):
            issues.append("German QA: repeated word")
        if '"' in target:
            issues.append("German QA: straight quotes used")
        if re.search(r"\b(the|and|with|for|from|your|our|you)\b", target, flags=re.IGNORECASE):
            issues.append("German QA: possible English word in German target")
        return issues

    @staticmethod
    def typography_issues(target, settings):
        issues = []
        if settings.flag_double_ellipsis and re.search(r"\.{4,}", target or ""):
            issues.append("Repeated ellipsis / too many dots")
        if settings.flag_double_dot and re.search(r"(?<!\.)\.\.(?!\.)", target or ""):
            issues.append("Double period detected")
        if settings.flag_double_spaces and re.search(r" {2,}", target or ""):
            issues.append("Double spaces detected")
        if settings.flag_space_before_period and re.search(r"\s+\.", target or ""):
            issues.append("Space before period detected")
        return issues

    @staticmethod
    def apply(records, settings, brand_rules, glossary_rules):
        for r in records:
            issues = []
            cats = []
            lqa_items = []
            s = r.source_text or ""
            t = r.target_text or ""

            if not t.strip():
                QAEngine.add_issue(issues, cats, lqa_items, "Missing Target", "Missing target", "Critical")

            if settings.flag_placeholder_issues and QAEngine.placeholders(s) != QAEngine.placeholders(t):
                QAEngine.add_issue(issues, cats, lqa_items, "Placeholders", "Placeholder mismatch", "Critical")

            if settings.flag_number_issues and QAEngine.numbers(s) != QAEngine.numbers(t):
                QAEngine.add_issue(issues, cats, lqa_items, "Numbers", "Number mismatch", "Critical")

            if settings.flag_tag_issues:
                if QAEngine.has_tag_issue(s):
                    QAEngine.add_issue(issues, cats, lqa_items, "Tags", "Source malformed tags", "Critical")
                if QAEngine.has_tag_issue(t):
                    QAEngine.add_issue(issues, cats, lqa_items, "Tags", "Target malformed tags", "Critical")

            if settings.flag_source_equals_target and s.strip() and t.strip() and s.strip() == t.strip():
                QAEngine.add_issue(issues, cats, lqa_items, "Source=Target", "Target equals source", "Major")

            if settings.flag_length_ratio and s:
                ratio = len(t) / max(1, len(s))
                if ratio < 0.35 or ratio > 2.8:
                    QAEngine.add_issue(issues, cats, lqa_items, "Length", "Suspicious length ratio", "Major")

            if settings.flag_brand_protection:
                for msg in QAEngine.brand_violations(s, t, brand_rules):
                    QAEngine.add_issue(issues, cats, lqa_items, "Brand Protection", msg, "Major")

            if settings.flag_glossary_violations:
                for msg in QAEngine.glossary_violations(s, t, glossary_rules):
                    QAEngine.add_issue(issues, cats, lqa_items, "Glossary", msg, "Major")

            if settings.flag_punctuation_issues and QAEngine.end_punct(s) != QAEngine.end_punct(t):
                if QAEngine.end_punct(s) or QAEngine.end_punct(t):
                    QAEngine.add_issue(issues, cats, lqa_items, "Punctuation", "Ending punctuation mismatch", "Minor")

            if settings.flag_german_micro_qa:
                for msg in QAEngine.german_micro_issues(r):
                    QAEngine.add_issue(issues, cats, lqa_items, "German QA", msg, "Minor")

            for msg in QAEngine.typography_issues(t, settings):
                QAEngine.add_issue(issues, cats, lqa_items, "Typography", msg, "Minor")

            r.issue_count = len(issues)
            r.severity = "Issues" if issues else "OK"
            r.issue_categories = "; ".join(sorted(set(cats)))
            r.issue_details = "; ".join(issues)

            if settings.enable_lqa_scoring:
                severities = [item[0] for item in lqa_items]
                r.lqa_severity = worst_lqa_severity(severities)
                r.lqa_penalty = sum(LQA_WEIGHTS.get(item[0], 0) for item in lqa_items)
                r.lqa_details = "; ".join([f"{sev}: {msg}" for sev, msg in lqa_items])
            else:
                r.lqa_severity = "OK" if not issues else "Unscored"
                r.lqa_penalty = 0
                r.lqa_details = ""


def _severity_issue_count(records, severity):
    prefix = f"{severity}:"
    return sum(1 for r in records for item in (r.lqa_details or "").split(";") if item.strip().startswith(prefix))


def build_stats(records):
    quality_score, total_penalty = calculate_quality_score(records)
    return {
        "total_segments": len(records),
        "segments_with_issues": sum(1 for r in records if r.issue_count),
        "clean_segments": sum(1 for r in records if not r.issue_count),
        "critical_issues": _severity_issue_count(records, "Critical"),
        "major_issues": _severity_issue_count(records, "Major"),
        "minor_issues": _severity_issue_count(records, "Minor"),
        "quality_score": quality_score,
        "quality_label": quality_label(quality_score),
        "total_lqa_penalty": total_penalty,
        "lqa_segment_severity": Counter(r.lqa_severity for r in records),
        "issue_categories": Counter(cat.strip() for r in records for cat in r.issue_categories.split(";") if cat.strip()),
        "file_types": Counter(r.file_type for r in records),
        "language_pairs": Counter(f"{r.source_lang}>{r.target_lang}" for r in records),
        "ai_reviewed_segments": sum(1 for r in records if r.ai_status not in {"", "Not reviewed"}),
        "ai_status_counts": Counter(r.ai_status for r in records if r.ai_status),
        "ai_severity_counts": Counter(r.ai_severity for r in records if r.ai_severity),
    }


def records_to_dataframe(records):
    import pandas as pd
    return pd.DataFrame([
        {
            "Record ID": r.record_id,
            "File": r.file_name,
            "Type": r.file_type,
            "Unit ID": r.unit_id,
            "Source Lang": r.source_lang,
            "Target Lang": r.target_lang,
            "Source": r.source_text,
            "Target": r.target_text,
            "Severity": r.severity,
            "Issue Count": r.issue_count,
            "Issue Categories": r.issue_categories,
            "Issue Details": r.issue_details,
            "LQA Severity": r.lqa_severity,
            "LQA Penalty": r.lqa_penalty,
            "LQA Details": r.lqa_details,
            "AI Status": r.ai_status,
            "AI Severity": r.ai_severity,
            "AI Suggestion": r.ai_suggestion,
            "AI Explanation": r.ai_explanation,
            "AI Model": r.ai_model,
            "AI Language": r.ai_language,
            "Repair Actions": r.repair_actions,
            "Notes": r.notes,
        }
        for r in records
    ])
