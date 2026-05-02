from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ============================================================
# DATA MODELS
# ============================================================

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

    ai_status: str = "Not reviewed"
    ai_severity: str = ""
    ai_suggestion: str = ""
    ai_explanation: str = ""
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


# ============================================================
# HELPERS
# ============================================================

def is_german(code: str) -> bool:
    return (code or "").lower().startswith("de")


def normalize_language_code(code: str) -> str:
    return (code or "").strip()


# ============================================================
# MARKUP CLEANER
# ============================================================

class MarkupCleaner:
    """
    Removes technical tags from visible text checks.

    Valid tag start:
    < immediately followed by letter or number

    Examples:
    <bpt>
    <x1>
    <ph id="1">

    Not a tag:
    < 5
    x < y
    A < B
    """

    REAL_TAG = re.compile(
        r"<[A-Za-z0-9][^>]*>",
        flags=re.DOTALL
    )

    ESCAPED_TAG = re.compile(
        r"&lt;[A-Za-z0-9].*?&gt;",
        flags=re.DOTALL
    )

    @staticmethod
    def strip_markup(text: str) -> str:
        value = text or ""

        value = MarkupCleaner.ESCAPED_TAG.sub(" ", value)
        value = MarkupCleaner.REAL_TAG.sub(" ", value)

        value = re.sub(r"\s+", " ", value).strip()

        return value


# ============================================================
# BRAND / GLOSSARY
# ============================================================

class BrandRules:
    def __init__(self):
        self.rules = []

    def load_from_dataframe(self, df):
        self.rules = []

        if df.shape[1] < 2:
            return 0

        for _, row in df.iterrows():
            s = str(row.iloc[0]).strip()
            t = str(row.iloc[1]).strip()

            if s and t:
                self.rules.append(
                    {"source": s, "required": t}
                )

        return len(self.rules)


class GlossaryRules:
    def __init__(self):
        self.rules = []

    def load_from_dataframe(self, df):
        self.rules = []

        if df.shape[1] < 2:
            return 0

        for _, row in df.iterrows():
            s = str(row.iloc[0]).strip()
            t = str(row.iloc[1]).strip()

            if s and t:
                self.rules.append(
                    {"source": s, "required": t}
                )

        return len(self.rules)


# ============================================================
# REPAIR ENGINE
# ============================================================

class RepairEngine:

    ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff]")

    @staticmethod
    def repair_text(text, settings):
        value = text or ""
        actions = []

        before = value
        if settings.normalize_unicode:
            value = unicodedata.normalize("NFC", value)
            if before != value:
                actions.append("Unicode normalized")

        before = value
        if settings.replace_nbsp:
            value = value.replace("\xa0", " ")
            if before != value:
                actions.append("NBSP replaced")

        before = value
        if settings.remove_zero_width:
            value = RepairEngine.ZERO_WIDTH.sub("", value)
            if before != value:
                actions.append("Zero-width removed")

        before = value
        if settings.collapse_spaces:
            value = re.sub(r"[ \t]{2,}", " ", value)
            if before != value:
                actions.append("Repeated spaces collapsed")

        before = value
        if settings.trim_spaces:
            value = value.strip()
            if before != value:
                actions.append("Trimmed")

        return value, actions

    @staticmethod
    def apply(records, settings):
        changed = 0

        for r in records:
            old = (r.source_text, r.target_text)

            r.source_text, a1 = RepairEngine.repair_text(
                r.source_text, settings
            )

            r.target_text, a2 = RepairEngine.repair_text(
                r.target_text, settings
            )

            r.repair_actions = "; ".join(
                [f"Source: {x}" for x in a1] +
                [f"Target: {x}" for x in a2]
            )

            new = (r.source_text, r.target_text)

            if old != new:
                changed += 1

        return changed


# ============================================================
# LQA
# ============================================================

LQA_WEIGHTS = {
    "Critical": 10,
    "Major": 5,
    "Minor": 1,
    "OK": 0
}

LQA_ORDER = {
    "OK": 0,
    "Minor": 1,
    "Major": 2,
    "Critical": 3
}


def worst(items):
    if not items:
        return "OK"

    return sorted(items, key=lambda x: LQA_ORDER[x])[-1]


def score_label(score):
    if score >= 90:
        return "Excellent"
    if score >= 80:
        return "Pass"
    if score >= 70:
        return "Review Required"
    return "Fail"


# ============================================================
# QA ENGINE
# ============================================================

class QAEngine:

    # strict tag opener only
    TAG_PATTERN = re.compile(
        r"</?[A-Za-z0-9][A-Za-z0-9:_-]*(?:\s[^>]*)?>"
    )

    @staticmethod
    def add_issue(issues, cats, lqa, cat, msg, sev):
        issues.append(msg)
        cats.append(cat)
        lqa.append((sev, msg))

    @staticmethod
    def placeholders(text):
        return re.findall(
            r"\{.*?\}|%s|%d",
            text or ""
        )

    @staticmethod
    def numbers(text):
        return re.findall(
            r"\d+(?:[\.,]\d+)?",
            text or ""
        )

    @staticmethod
    def end_punct(text):
        val = (text or "").strip()
        return val[-1] if val and val[-1] in ".,:;!?" else ""

    @staticmethod
    def malformed_tag(text):
        txt = text or ""

        tags = QAEngine.TAG_PATTERN.findall(txt)

        if not tags:
            return False

        opens = sum(
            1 for t in tags
            if not t.startswith("</")
            and not t.endswith("/>")
        )

        closes = sum(
            1 for t in tags
            if t.startswith("</")
        )

        return opens != closes

    @staticmethod
    def brand_violations(source, target, rules):
        out = []

        for r in rules.rules:
            patt = r"\b" + re.escape(r["source"]) + r"\b"

            if re.search(patt, source, re.I):
                req = r"\b" + re.escape(r["required"]) + r"\b"

                if not re.search(req, target, re.I):
                    out.append(
                        f"Protected term '{r['source']}' should be '{r['required']}'"
                    )

        return out

    @staticmethod
    def glossary_violations(source, target, rules):
        out = []

        for r in rules.rules:
            patt = r"\b" + re.escape(r["source"]) + r"\b"

            if re.search(patt, source, re.I):
                req = r"\b" + re.escape(r["required"]) + r"\b"

                if not re.search(req, target, re.I):
                    out.append(
                        f"Glossary violation: {r['source']} -> {r['required']}"
                    )

        return out

    @staticmethod
    def german_micro(record):
        if not is_german(record.target_lang):
            return []

        visible = MarkupCleaner.strip_markup(
            record.target_text
        )

        issues = []

        if '"' in visible:
            issues.append(
                "German QA: straight quotes used"
            )

        if re.search(
            r"\b(\w+)\s+\1\b",
            visible,
            re.I
        ):
            issues.append(
                "German QA: repeated word"
            )

        return issues

    @staticmethod
    def typography(text, settings):
        visible = MarkupCleaner.strip_markup(text)

        out = []

        if settings.flag_double_ellipsis:
            if re.search(r"\.{4,}", visible):
                out.append(
                    "Repeated ellipsis / too many dots"
                )

        if settings.flag_double_dot:
            if re.search(
                r"(?<!\.)\.\.(?!\.)",
                visible
            ):
                out.append(
                    "Double period detected"
                )

        if settings.flag_double_spaces:
            if re.search(
                r" {2,}",
                visible
            ):
                out.append(
                    "Double spaces detected"
                )

        if settings.flag_space_before_period:
            if re.search(
                r"\s+\.",
                visible
            ):
                out.append(
                    "Space before period detected"
                )

        return out

    @staticmethod
    def apply(records, settings, brand_rules, glossary_rules):

        for r in records:

            issues = []
            cats = []
            lqa = []

            s = r.source_text or ""
            t = r.target_text or ""

            # -------------------
            # Critical
            # -------------------

            if not t.strip():
                QAEngine.add_issue(
                    issues, cats, lqa,
                    "Missing Target",
                    "Missing target",
                    "Critical"
                )

            if settings.flag_placeholder_issues:
                if QAEngine.placeholders(s) != QAEngine.placeholders(t):
                    QAEngine.add_issue(
                        issues, cats, lqa,
                        "Placeholders",
                        "Placeholder mismatch",
                        "Critical"
                    )

            if settings.flag_number_issues:
                if QAEngine.numbers(s) != QAEngine.numbers(t):
                    QAEngine.add_issue(
                        issues, cats, lqa,
                        "Numbers",
                        "Number mismatch",
                        "Critical"
                    )

            if settings.flag_tag_issues:
                if QAEngine.malformed_tag(s):
                    QAEngine.add_issue(
                        issues, cats, lqa,
                        "Tags",
                        "Source malformed tags",
                        "Critical"
                    )

                if QAEngine.malformed_tag(t):
                    QAEngine.add_issue(
                        issues, cats, lqa,
                        "Tags",
                        "Target malformed tags",
                        "Critical"
                    )

            # -------------------
            # Major
            # -------------------

            if settings.flag_source_equals_target:
                if s.strip() and t.strip() and s.strip() == t.strip():
                    QAEngine.add_issue(
                        issues, cats, lqa,
                        "Source=Target",
                        "Target equals source",
                        "Major"
                    )

            if settings.flag_brand_protection:
                for x in QAEngine.brand_violations(
                    s, t, brand_rules
                ):
                    QAEngine.add_issue(
                        issues, cats, lqa,
                        "Brand Protection",
                        x,
                        "Major"
                    )

            if settings.flag_glossary_violations:
                for x in QAEngine.glossary_violations(
                    s, t, glossary_rules
                ):
                    QAEngine.add_issue(
                        issues, cats, lqa,
                        "Glossary",
                        x,
                        "Major"
                    )

            # -------------------
            # Minor
            # -------------------

            if settings.flag_punctuation_issues:
                if QAEngine.end_punct(s) != QAEngine.end_punct(t):
                    QAEngine.add_issue(
                        issues, cats, lqa,
                        "Punctuation",
                        "Ending punctuation mismatch",
                        "Minor"
                    )

            if settings.flag_german_micro_qa:
                for x in QAEngine.german_micro(r):
                    QAEngine.add_issue(
                        issues, cats, lqa,
                        "German QA",
                        x,
                        "Minor"
                    )

            for x in QAEngine.typography(t, settings):
                QAEngine.add_issue(
                    issues, cats, lqa,
                    "Typography",
                    x,
                    "Minor"
                )

            # Finalize
            r.issue_count = len(issues)
            r.severity = "Issues" if issues else "OK"
            r.issue_categories = "; ".join(
                sorted(set(cats))
            )
            r.issue_details = "; ".join(issues)

            sev = [x[0] for x in lqa]

            r.lqa_severity = worst(sev)
            r.lqa_penalty = sum(
                LQA_WEIGHTS[x[0]] for x in lqa
            )
            r.lqa_details = "; ".join(
                f"{a}: {b}" for a, b in lqa
            )


# ============================================================
# STATS
# ============================================================

def build_stats(records):

    penalty = sum(r.lqa_penalty for r in records)

    score = max(0, 100 - penalty)

    return {
        "total_segments": len(records),
        "segments_with_issues": sum(
            1 for r in records if r.issue_count
        ),
        "clean_segments": sum(
            1 for r in records if not r.issue_count
        ),
        "quality_score": score,
        "quality_label": score_label(score),
        "issue_categories": Counter(
            cat.strip()
            for r in records
            for cat in r.issue_categories.split(";")
            if cat.strip()
        ),
    }


# ============================================================
# DATAFRAME
# ============================================================

def records_to_dataframe(records):
    import pandas as pd

    rows = []

    for r in records:
        rows.append({
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
        })

    return pd.DataFrame(rows)
