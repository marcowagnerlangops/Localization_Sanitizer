# app.py
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import pandas as pd
import streamlit as st

from adapters import read_any, write_by_type
from exporters import build_xlsx_report
from ai_reviewer import (
    AI_REVIEW_MODES,
    LANGUAGE_OPTIONS,
    STRICTNESS_OPTIONS,
    apply_ai_result,
    review_segment_with_openai,
    select_records_for_ai,
)
from sanitizer_core import (
    BrandRules,
    GlossaryRules,
    QAEngine,
    RepairEngine,
    SanitizerSettings,
    build_stats,
    records_to_dataframe,
)

APP_TITLE = "LangOps Sanitizer Pro"
APP_VERSION = "4.1"
MAKER_LINE = "Made by LangOps Solutions"


# ==========================================================
# STATE
# ==========================================================

def init_state():
    if "records" not in st.session_state:
        st.session_state.records = []
    if "stats" not in st.session_state:
        st.session_state.stats = {}
    if "brand_rules" not in st.session_state:
        st.session_state.brand_rules = BrandRules()
    if "glossary_rules" not in st.session_state:
        st.session_state.glossary_rules = GlossaryRules()
    if "logs" not in st.session_state:
        st.session_state.logs = []


def log(msg):
    stamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{stamp}] {msg}")


# ==========================================================
# SIDEBAR
# ==========================================================

def sidebar_settings():
    st.sidebar.header("Sanitizer Settings")

    st.sidebar.subheader("Safe Auto Repairs")
    normalize_unicode = st.sidebar.checkbox("Normalize Unicode", True)
    trim_spaces = st.sidebar.checkbox("Trim Spaces", True)
    remove_zero_width = st.sidebar.checkbox("Remove Zero Width", True)
    replace_nbsp = st.sidebar.checkbox("Replace NBSP", True)
    collapse_spaces = st.sidebar.checkbox("Collapse Repeated Spaces", True)
    normalize_language_codes = st.sidebar.checkbox("Normalize Language Codes", True)

    st.sidebar.subheader("QA Checks")
    flag_tag_issues = st.sidebar.checkbox("Malformed Tags", True)
    flag_source_equals_target = st.sidebar.checkbox("Source = Target", True)
    flag_german_micro_qa = st.sidebar.checkbox("German Micro QA", False)
    flag_brand_protection = st.sidebar.checkbox("Brand Protection", True)
    flag_placeholder_issues = st.sidebar.checkbox("Placeholder Mismatch", True)
    flag_number_issues = st.sidebar.checkbox("Number Mismatch", True)
    flag_punctuation_issues = st.sidebar.checkbox("Punctuation Mismatch", True)
    flag_length_ratio = st.sidebar.checkbox("Suspicious Length Ratio", True)
    flag_double_ellipsis = st.sidebar.checkbox("Repeated Ellipsis / ....", True)
    flag_double_spaces = st.sidebar.checkbox("Double Spaces", True)
    flag_double_dot = st.sidebar.checkbox("Double Dot ..", True)
    flag_space_before_period = st.sidebar.checkbox("Space Before Period", True)
    flag_glossary_violations = st.sidebar.checkbox("Glossary Violations", True)

    st.sidebar.subheader("LQA")
    enable_lqa_scoring = st.sidebar.checkbox("Enable LQA Severity Scoring", True)

    return SanitizerSettings(
        normalize_unicode=normalize_unicode,
        trim_spaces=trim_spaces,
        remove_zero_width=remove_zero_width,
        replace_nbsp=replace_nbsp,
        collapse_spaces=collapse_spaces,
        normalize_language_codes=normalize_language_codes,
        flag_tag_issues=flag_tag_issues,
        flag_source_equals_target=flag_source_equals_target,
        flag_german_micro_qa=flag_german_micro_qa,
        flag_brand_protection=flag_brand_protection,
        flag_placeholder_issues=flag_placeholder_issues,
        flag_number_issues=flag_number_issues,
        flag_punctuation_issues=flag_punctuation_issues,
        flag_length_ratio=flag_length_ratio,
        flag_double_ellipsis=flag_double_ellipsis,
        flag_double_spaces=flag_double_spaces,
        flag_double_dot=flag_double_dot,
        flag_space_before_period=flag_space_before_period,
        flag_glossary_violations=flag_glossary_violations,
        enable_lqa_scoring=enable_lqa_scoring,
    )


# ==========================================================
# HELPERS
# ==========================================================

def rerun_qa(settings):
    QAEngine.apply(
        st.session_state.records,
        settings,
        st.session_state.brand_rules,
        st.session_state.glossary_rules,
    )
    st.session_state.stats = build_stats(st.session_state.records)


def parse_uploaded_files(uploaded_files, src_lang, tgt_lang):
    all_records = []
    next_id = 1

    for file in uploaded_files:
        records, meta = read_any(file, next_id, src_lang, tgt_lang)
        all_records.extend(records)

        if records:
            next_id = max(x.record_id for x in all_records) + 1

        log(f"Loaded {file.name}: {len(records)} segments")

    return all_records


def dedupe_records(records, mode):
    if mode == "No Deduplication":
        return records

    result = []
    seen = set()

    for r in records:
        if mode == "Source + Target":
            key = (
                r.source_lang,
                r.target_lang,
                r.source_text.strip(),
                r.target_text.strip(),
            )
        elif mode == "Source Only":
            key = (r.source_lang, r.source_text.strip())
        else:
            key = r.record_id

        if key not in seen:
            seen.add(key)
            result.append(r)

    return result


def filter_dataframe(df):
    if df.empty:
        return df

    with st.expander("Filters", expanded=True):
        c1, c2, c3 = st.columns(3)

        with c1:
            severity = st.selectbox("Severity", ["All", "Issues", "OK"])

        with c2:
            lqa = st.selectbox(
                "LQA Severity",
                ["All", "Critical", "Major", "Minor", "OK", "Unscored"],
            )

        with c3:
            file_type = st.selectbox(
                "File Type",
                ["All"] + sorted(df["Type"].dropna().unique().tolist()),
            )

        search = st.text_input("Search")

    out = df.copy()

    if severity != "All":
        out = out[out["Severity"] == severity]

    if lqa != "All":
        out = out[out["LQA Severity"] == lqa]

    if file_type != "All":
        out = out[out["Type"] == file_type]

    if search.strip():
        needle = search.lower()
        out = out[
            out["Source"].astype(str).str.lower().str.contains(needle, regex=False)
            | out["Target"].astype(str).str.lower().str.contains(needle, regex=False)
            | out["Issue Details"].astype(str).str.lower().str.contains(needle, regex=False)
        ]

    return out


# ==========================================================
# MAIN
# ==========================================================

def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🧼", layout="wide")

    init_state()
    settings = sidebar_settings()

    st.title("🧼 LangOps Sanitizer Pro")
    st.caption("Clean, QA, merge, edit and export localization assets.")

    tabs = st.tabs([
        "Upload & Analyze",
        "Dashboard",
        "Segments",
        "Review & Edit",
        "AI Review",
        "Merge Center",
        "Export",
        "Logs",
    ])

    # ======================================================
    # TAB 1 Upload
    # ======================================================

    with tabs[0]:
        uploaded = st.file_uploader(
            "Upload files",
            type=["tmx", "xlsx", "csv", "xlf", "xliff", "txlf", "xlz"],
            accept_multiple_files=True,
        )

        c1, c2 = st.columns(2)

        with c1:
            src_lang = st.text_input("Default Source Language", "en-US")

        with c2:
            tgt_lang = st.text_input("Default Target Language", "de-DE")

        if st.button("Analyze Files", use_container_width=True):
            if uploaded:
                st.session_state.records = parse_uploaded_files(uploaded, src_lang, tgt_lang)
                RepairEngine.apply(st.session_state.records, settings)
                rerun_qa(settings)
                st.success("Analysis complete.")

    # ======================================================
    # TAB 2 Dashboard
    # ======================================================

    with tabs[1]:
        if not st.session_state.records:
            st.info("No project loaded.")
        else:
            stats = st.session_state.stats

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Quality Score", stats.get("quality_score", 100))
            c2.metric("Segments", stats.get("total_segments", 0))
            c3.metric("Issues", stats.get("segments_with_issues", 0))
            c4.metric("Clean", stats.get("clean_segments", 0))

    # ======================================================
    # TAB 3 Segments
    # ======================================================

    with tabs[2]:
        if not st.session_state.records:
            st.info("No records loaded.")
        else:
            df = records_to_dataframe(st.session_state.records)
            filtered = filter_dataframe(df)
            st.dataframe(filtered, use_container_width=True, height=720)

    # ======================================================
    # TAB 4 Review & Edit
    # ======================================================

    with tabs[3]:
        if not st.session_state.records:
            st.info("No records loaded.")
        else:
            st.subheader("Review & Edit")

            review_mode = st.selectbox(
                "Queue",
                [
                    "All Issues",
                    "Critical",
                    "Major",
                    "Minor",
                    "AI Suggestions",
                    "All Segments",
                ],
            )

            records = st.session_state.records

            if review_mode == "All Issues":
                review_records = [r for r in records if r.issue_count > 0]
            elif review_mode in {"Critical", "Major", "Minor"}:
                review_records = [r for r in records if r.lqa_severity == review_mode]
            elif review_mode == "AI Suggestions":
                review_records = [r for r in records if r.ai_suggestion]
            else:
                review_records = records

            if review_records:
                selected_id = st.selectbox(
                    "Select Segment",
                    [r.record_id for r in review_records],
                )

                record = next(r for r in records if r.record_id == selected_id)

                st.text_area("Source", value=record.source_text, height=130, disabled=True)

                edited_target = st.text_area(
                    "Target",
                    value=record.target_text,
                    height=180,
                )

                notes = st.text_area(
                    "Notes",
                    value=record.notes,
                    height=80,
                )

                c1, c2, c3 = st.columns(3)

                with c1:
                    if st.button("Save Changes", use_container_width=True):
                        record.target_text = edited_target
                        record.notes = notes
                        rerun_qa(settings)
                        st.success("Saved.")
                        st.rerun()

                with c2:
                    if st.button(
                        "Apply AI Suggestion",
                        use_container_width=True,
                        disabled=not bool(record.ai_suggestion),
                    ):
                        record.target_text = record.ai_suggestion
                        rerun_qa(settings)
                        st.success("AI suggestion applied.")
                        st.rerun()

                with c3:
                    if st.button("Mark Reviewed", use_container_width=True):
                        record.notes = (record.notes + "; " if record.notes else "") + "Reviewed"
                        st.success("Marked reviewed.")
                        st.rerun()

                st.divider()
                st.write("**Issues:**", record.issue_details or "-")
                st.write("**LQA:**", record.lqa_details or "-")

            else:
                st.success("No records in selected queue.")

    # ======================================================
    # TAB 5 AI Review
    # ======================================================

    with tabs[4]:
        if not st.session_state.records:
            st.info("Load files first.")
        else:
            st.subheader("AI Review")

            api_key = st.text_input("OpenAI API Key", type="password")
            model = st.text_input("Model", "gpt-5.2")
            language = st.selectbox("Language", LANGUAGE_OPTIONS)
            strictness = st.selectbox("Strictness", STRICTNESS_OPTIONS)
            mode = st.selectbox("Scope", AI_REVIEW_MODES)

            if st.button("Run AI Review", use_container_width=True):
                selected = select_records_for_ai(
                    st.session_state.records,
                    mode,
                    25,
                )

                for r in selected:
                    try:
                        result = review_segment_with_openai(
                            record=r,
                            api_key=api_key,
                            model=model,
                            target_language=language,
                            strictness=strictness,
                            custom_instructions="",
                        )
                        apply_ai_result(r, result)
                    except Exception as exc:
                        r.ai_status = "ERROR"
                        r.ai_explanation = str(exc)

                rerun_qa(settings)
                st.success("AI review complete.")

    # ======================================================
    # TAB 6 Merge
    # ======================================================

    with tabs[5]:
        if st.session_state.records:
            dedupe = st.selectbox(
                "Deduplication",
                ["No Deduplication", "Source + Target", "Source Only"],
            )

            export_type = st.selectbox(
                "Export Format",
                ["tmx", "xlsx", "csv", "xliff"],
            )

            merged = dedupe_records(st.session_state.records, dedupe)

            data, name, mime = write_by_type(merged, export_type)

            st.download_button(
                f"Download {export_type.upper()}",
                data=data,
                file_name=name,
                mime=mime,
                use_container_width=True,
            )

    # ======================================================
    # TAB 7 Export
    # ======================================================

    with tabs[6]:
        if st.session_state.records:
            report = build_xlsx_report(
                st.session_state.records,
                st.session_state.stats,
            )

            st.download_button(
                "Download XLSX QA Report",
                data=report,
                file_name="langops_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    # ======================================================
    # TAB 8 Logs
    # ======================================================

    with tabs[7]:
        st.text_area(
            "Logs",
            "\n".join(st.session_state.logs),
            height=720,
        )


if __name__ == "__main__":
    main()
