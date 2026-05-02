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
    build_ai_prompt,
    select_records_for_ai,
)

from ai_providers import (
    review_openai,
    review_ollama,
    review_azure,
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

APP_TITLE = "LangOps Sanitizer"
APP_VERSION = "1.2"
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

    st.sidebar.subheader("LQA Scoring")
    enable_lqa_scoring = st.sidebar.checkbox("Enable LQA Severity Scoring", True)
    st.sidebar.caption("Score = 100 - Critical×10 - Major×5 - Minor×1")

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
            key = (
                r.source_lang,
                r.source_text.strip(),
            )
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
            lqa_severity = st.selectbox(
                "LQA Severity",
                ["All", "Critical", "Major", "Minor", "OK", "Unscored"],
            )

        with c3:
            file_type = st.selectbox(
                "File Type",
                ["All"] + sorted(df["Type"].dropna().unique().tolist()),
            )

        c4, c5, c6 = st.columns(3)

        with c4:
            search = st.text_input("Search")

        with c5:
            categories = sorted({
                cat.strip()
                for val in df["Issue Categories"].fillna("")
                for cat in str(val).split(";")
                if cat.strip()
            })
            category = st.selectbox("Issue Category", ["All"] + categories)

        with c6:
            if "AI Status" in df.columns:
                ai_status_values = ["All"] + sorted(
                    df["AI Status"].fillna("Not reviewed").unique().tolist()
                )
            else:
                ai_status_values = ["All"]

            ai_status = st.selectbox("AI Status", ai_status_values)

    out = df.copy()

    if severity != "All":
        out = out[out["Severity"] == severity]

    if lqa_severity != "All":
        out = out[out["LQA Severity"] == lqa_severity]

    if file_type != "All":
        out = out[out["Type"] == file_type]

    if category != "All":
        out = out[
            out["Issue Categories"]
            .fillna("")
            .str.contains(category, case=False, regex=False)
        ]

    if ai_status != "All" and "AI Status" in out.columns:
        out = out[out["AI Status"].fillna("Not reviewed") == ai_status]

    if search.strip():
        needle = search.lower()

        ai_suggestion = (
            out["AI Suggestion"].astype(str).str.lower().str.contains(needle, regex=False)
            if "AI Suggestion" in out.columns
            else False
        )

        ai_explanation = (
            out["AI Explanation"].astype(str).str.lower().str.contains(needle, regex=False)
            if "AI Explanation" in out.columns
            else False
        )

        out = out[
            out["Source"].astype(str).str.lower().str.contains(needle, regex=False)
            | out["Target"].astype(str).str.lower().str.contains(needle, regex=False)
            | out["Issue Details"].astype(str).str.lower().str.contains(needle, regex=False)
            | out["LQA Details"].astype(str).str.lower().str.contains(needle, regex=False)
            | ai_suggestion
            | ai_explanation
        ]

    return out


# ==========================================================
# MAIN
# ==========================================================

def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🧼", layout="wide")
    init_state()

    st.title("🧼 LangOps Sanitizer")
    st.caption("Clean, QA, merge, edit and export TMX, XLSX, CSV, XLIFF, XLF, TXLF and XLZ localization assets.")

    settings = sidebar_settings()

    # ======================================================
    # SIDEBAR UPLOADS
    # ======================================================

    st.sidebar.divider()
    st.sidebar.subheader("Brand Protection")

    brand_file = st.sidebar.file_uploader(
        "Upload Brand Rules XLSX / CSV",
        type=["xlsx", "csv"],
        key="brand",
    )

    if brand_file:
        try:
            if brand_file.name.lower().endswith(".csv"):
                df = pd.read_csv(brand_file, header=None)
            else:
                df = pd.read_excel(brand_file, header=None)

            count = st.session_state.brand_rules.load_from_dataframe(df)
            st.sidebar.success(f"{count} brand rules loaded")

        except Exception as exc:
            st.sidebar.error(str(exc))

    st.sidebar.divider()
    st.sidebar.subheader("Glossary")

    glossary_file = st.sidebar.file_uploader(
        "Upload Glossary XLSX / CSV",
        type=["xlsx", "csv"],
        key="glossary",
    )

    if glossary_file:
        try:
            if glossary_file.name.lower().endswith(".csv"):
                df = pd.read_csv(glossary_file, header=None)
            else:
                df = pd.read_excel(glossary_file, header=None)

            count = st.session_state.glossary_rules.load_from_dataframe(df)
            st.sidebar.success(f"{count} glossary terms loaded")

        except Exception as exc:
            st.sidebar.error(str(exc))

    st.sidebar.divider()
    st.sidebar.caption(f"{MAKER_LINE} · v{APP_VERSION}")

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
    # TAB 1 UPLOAD
    # ======================================================

    with tabs[0]:
        st.subheader("Upload Files")

        uploaded = st.file_uploader(
            "Upload one or multiple files",
            type=["tmx", "xlsx", "csv", "xlf", "xliff", "txlf", "xlz"],
            accept_multiple_files=True,
        )

        c1, c2 = st.columns(2)

        with c1:
            src_lang = st.text_input("Default Source Language", "en-US")

        with c2:
            tgt_lang = st.text_input("Default Target Language", "de-DE")

        c1, c2, c3 = st.columns(3)

        with c1:
            if st.button("Analyze Files", use_container_width=True):
                if not uploaded:
                    st.warning("Upload files first.")
                else:
                    try:
                        with st.spinner("Analyzing..."):
                            st.session_state.records = parse_uploaded_files(
                                uploaded,
                                src_lang,
                                tgt_lang,
                            )

                            changed = RepairEngine.apply(
                                st.session_state.records,
                                settings,
                            )

                            rerun_qa(settings)

                            log(
                                f"Analysis complete | "
                                f"{len(st.session_state.records)} segments | "
                                f"{changed} repaired"
                            )

                        st.success("Analysis complete.")

                    except Exception as exc:
                        st.error(str(exc))

        with c2:
            if st.button("Run Auto Repairs", use_container_width=True):
                changed = RepairEngine.apply(
                    st.session_state.records,
                    settings,
                )

                rerun_qa(settings)

                log(f"Auto repairs run | {changed} segments updated")
                st.success(f"{changed} segments updated")

        with c3:
            if st.button("Clear Project", use_container_width=True):
                st.session_state.records = []
                st.session_state.stats = {}
                st.session_state.logs = []
                st.session_state.brand_rules = BrandRules()
                st.session_state.glossary_rules = GlossaryRules()

                st.success("Project cleared")
                st.rerun()

    # ======================================================
    # TAB 2 DASHBOARD
    # ======================================================

    with tabs[1]:
        if not st.session_state.records:
            st.info("No project loaded.")
        else:
            stats = st.session_state.stats

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Quality Score", f"{stats.get('quality_score', 100)} / 100")
            c2.metric("Quality Label", stats.get("quality_label", "Excellent"))
            c3.metric("Segments", stats.get("total_segments", 0))
            c4.metric("Issues", stats.get("segments_with_issues", 0))

            c5, c6, c7, c8 = st.columns(4)

            c5.metric("Critical", stats.get("critical_issues", 0))
            c6.metric("Major", stats.get("major_issues", 0))
            c7.metric("Minor", stats.get("minor_issues", 0))
            c8.metric("Penalty", stats.get("total_lqa_penalty", 0))

            c9, c10, c11, c12 = st.columns(4)

            c9.metric("Clean", stats.get("clean_segments", 0))
            c10.metric("Glossary Terms", len(st.session_state.glossary_rules.rules))
            c11.metric("Brand Rules", len(st.session_state.brand_rules.rules))

            ai_reviewed = stats.get("ai_reviewed_segments", 0)
            c12.metric("AI Reviewed", ai_reviewed)

            ai_status_counts = stats.get("ai_status_counts", {})
            ai_pass = ai_status_counts.get("PASS", 0)
            ai_review = ai_status_counts.get("REVIEW", 0)
            ai_error = ai_status_counts.get("ERROR", 0)

            c13, c14, c15 = st.columns(3)
            c13.metric("AI PASS", ai_pass)
            c14.metric("AI REVIEW", ai_review)
            c15.metric("AI ERROR", ai_error)

            st.divider()

            col1, col2 = st.columns(2)

            with col1:
                if stats.get("issue_categories"):
                    st.subheader("Issue Categories")
                    chart = pd.DataFrame(
                        stats["issue_categories"].items(),
                        columns=["Category", "Count"],
                    )
                    st.bar_chart(chart, x="Category", y="Count")

            with col2:
                if stats.get("lqa_segment_severity"):
                    st.subheader("LQA Segment Severity")
                    sev_chart = pd.DataFrame(
                        stats["lqa_segment_severity"].items(),
                        columns=["Severity", "Count"],
                    )
                    st.bar_chart(sev_chart, x="Severity", y="Count")

            st.subheader("LQA Scoring Model")
            st.info(
                "Quality Score = 100 - Critical×10 - Major×5 - Minor×1. "
                "Critical includes missing targets, placeholder mismatches, number mismatches and malformed tags. "
                "Major includes glossary, brand, source=target and severe length-ratio issues. "
                "Minor includes typography, punctuation and German micro-QA issues."
            )

    # ======================================================
    # TAB 3 SEGMENTS
    # ======================================================

    with tabs[2]:
        if not st.session_state.records:
            st.info("No records loaded.")
        else:
            df = records_to_dataframe(st.session_state.records)
            filtered = filter_dataframe(df)

            st.caption(f"Showing {len(filtered)} of {len(df)} records.")

            st.dataframe(
                filtered,
                use_container_width=True,
                hide_index=True,
                height=720,
            )

    # ======================================================
    # TAB 4 REVIEW & EDIT
    # ======================================================

    with tabs[3]:
        if not st.session_state.records:
            st.info("No records loaded.")
        else:
            st.subheader("Review & Edit")

            st.write(
                "Review problematic segments, edit the target text, apply AI suggestions, "
                "and re-run QA immediately."
            )

            review_mode = st.selectbox(
                "Review Queue",
                [
                    "All Issues",
                    "Critical",
                    "Major",
                    "Minor",
                    "Glossary",
                    "Brand Protection",
                    "Typography",
                    "AI Suggestions",
                    "All Segments",
                ],
            )

            records = st.session_state.records

            if review_mode == "All Issues":
                review_records = [r for r in records if r.issue_count > 0]

            elif review_mode in {"Critical", "Major", "Minor"}:
                review_records = [
                    r for r in records
                    if r.lqa_severity == review_mode
                ]

            elif review_mode == "Glossary":
                review_records = [
                    r for r in records
                    if "Glossary" in (r.issue_categories or "")
                ]

            elif review_mode == "Brand Protection":
                review_records = [
                    r for r in records
                    if "Brand Protection" in (r.issue_categories or "")
                ]

            elif review_mode == "Typography":
                review_records = [
                    r for r in records
                    if "Typography" in (r.issue_categories or "")
                ]

            elif review_mode == "AI Suggestions":
                review_records = [
                    r for r in records
                    if getattr(r, "ai_suggestion", "")
                ]

            else:
                review_records = records

            if not review_records:
                st.success("No records found for the selected review queue.")
            else:
                st.info(f"{len(review_records)} segment(s) in this review queue.")

                selected_id = st.selectbox(
                    "Select Segment",
                    [r.record_id for r in review_records],
                    format_func=lambda rid: (
                        f"Record {rid} | "
                        f"{next(r for r in review_records if r.record_id == rid).lqa_severity} | "
                        f"{next(r for r in review_records if r.record_id == rid).issue_categories}"
                    ),
                )

                record = next(
                    r for r in st.session_state.records
                    if r.record_id == selected_id
                )

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Record ID", record.record_id)
                c2.metric("LQA Severity", record.lqa_severity)
                c3.metric("Penalty", record.lqa_penalty)
                c4.metric("Issue Count", record.issue_count)

                st.divider()

                st.caption(
                    f"File: {record.file_name} | "
                    f"Type: {record.file_type} | "
                    f"Unit ID: {record.unit_id}"
                )

                with st.expander("Issue Details", expanded=True):
                    st.write(record.issue_details or "No issues.")

                    if record.lqa_details:
                        st.write("**LQA Details:**")
                        st.write(record.lqa_details)

                if getattr(record, "ai_suggestion", ""):
                    with st.expander("AI Suggestion", expanded=True):
                        st.write("**AI Status:**", record.ai_status)
                        st.write("**AI Severity:**", record.ai_severity or "-")
                        st.write("**AI Explanation:**", record.ai_explanation or "-")
                        st.text_area(
                            "AI Suggestion Text",
                            value=record.ai_suggestion,
                            height=120,
                            disabled=True,
                        )

                with st.form("edit_segment_form"):
                    st.text_area(
                        "Source",
                        value=record.source_text,
                        height=150,
                        disabled=True,
                    )

                    edited_target = st.text_area(
                        "Target",
                        value=record.target_text,
                        height=180,
                    )

                    edited_notes = st.text_area(
                        "Notes",
                        value=record.notes,
                        height=80,
                    )

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        save_clicked = st.form_submit_button(
                            "Save Changes",
                            use_container_width=True,
                        )

                    with c2:
                        apply_ai_clicked = st.form_submit_button(
                            "Apply AI Suggestion",
                            use_container_width=True,
                            disabled=not bool(getattr(record, "ai_suggestion", "")),
                        )

                    with c3:
                        mark_reviewed_clicked = st.form_submit_button(
                            "Mark Reviewed",
                            use_container_width=True,
                        )

                if save_clicked:
                    record.target_text = edited_target
                    record.notes = edited_notes
                    rerun_qa(settings)
                    log(f"Edited record {record.record_id}")
                    st.success("Changes saved and QA re-run.")
                    st.rerun()

                if apply_ai_clicked and getattr(record, "ai_suggestion", ""):
                    record.target_text = record.ai_suggestion
                    record.notes = (
                        (record.notes + "; ") if record.notes else ""
                    ) + "AI suggestion accepted"
                    rerun_qa(settings)
                    log(f"Applied AI suggestion for record {record.record_id}")
                    st.success("AI suggestion applied and QA re-run.")
                    st.rerun()

                if mark_reviewed_clicked:
                    record.notes = (
                        (record.notes + "; ") if record.notes else ""
                    ) + "Reviewed"
                    log(f"Marked record {record.record_id} as reviewed")
                    st.success("Record marked as reviewed.")
                    st.rerun()

    # ======================================================
    # TAB 5 AI REVIEW
    # ======================================================
    
    with tabs[4]:
        if not st.session_state.records:
            st.info("Load and analyze files first.")
        else:
            st.subheader("AI Language Review")
    
            st.write(
                "Use OpenAI, Ollama or Azure OpenAI for grammar, fluency, idiomatic usage and rewrite suggestions."
            )
    
            c1, c2, c3 = st.columns(3)
    
            with c1:
                provider = st.selectbox(
                    "AI Provider",
                    ["OpenAI", "Ollama", "Azure OpenAI"]
                )
            
                if provider == "OpenAI":
                    default_model = "gpt-5.2"
                elif provider == "Ollama":
                    default_model = "qwen2.5:7b"
                else:
                    default_model = "your-deployment-name"
            
                model = st.text_input(
                    "Model / Deployment",
                    value=default_model
                )
    
            with c2:
                language = st.selectbox(
                    "Target Language for AI Review",
                    LANGUAGE_OPTIONS,
                    index=0,
                )
    
                custom_language = ""
    
                if language == "Custom":
                    custom_language = st.text_input(
                        "Custom Target Language"
                    )
    
                strictness = st.selectbox(
                    "Review Strictness",
                    STRICTNESS_OPTIONS,
                    index=1,
                )
    
            with c3:
                mode = st.selectbox(
                    "Review Scope",
                    AI_REVIEW_MODES,
                    index=0,
                )
    
                max_segments = st.number_input(
                    "Max Segments per Run",
                    min_value=1,
                    max_value=500,
                    value=25,
                    step=5,
                )
    
            # Provider Credentials
            if provider == "OpenAI":
                api_key = st.text_input(
                    "OpenAI API Key",
                    type="password"
                )
                base_url = ""
                endpoint = ""
    
            elif provider == "Ollama":
                api_key = ""
                base_url = st.text_input(
                    "Ollama Base URL",
                    value="http://localhost:11434"
                )
                endpoint = ""
    
            else:
                endpoint = st.text_input(
                    "Azure Endpoint",
                    placeholder="https://your-resource.openai.azure.com"
                )
                api_key = st.text_input(
                    "Azure API Key",
                    type="password"
                )
                base_url = ""
    
            custom_instructions = st.text_area(
                "Optional Custom AI Instructions",
                value="",
                height=90,
            )
    
            target_language = (
                custom_language.strip()
                if language == "Custom" and custom_language.strip()
                else language
            )
    
            selected_for_ai = select_records_for_ai(
                st.session_state.records,
                mode,
                int(max_segments),
            )
    
            st.info(
                f"Segments selected for AI review: {len(selected_for_ai)}"
            )
    
            if st.button(
                "Run AI Review",
                type="primary",
                use_container_width=True
            ):
                if not selected_for_ai:
                    st.warning("No segments selected.")
                else:
                    progress = st.progress(0)
                    reviewed = 0
                    errors = 0
    
                    for idx, record in enumerate(
                        selected_for_ai,
                        start=1
                    ):
                        try:
                            prompt = build_ai_prompt(
                                record,
                                target_language,
                                strictness,
                                custom_instructions,
                            )
    
                            if provider == "OpenAI":
                                result = review_openai(
                                    prompt,
                                    api_key,
                                    model,
                                    target_language
                                )
    
                            elif provider == "Ollama":
                                result = review_ollama(
                                    prompt,
                                    base_url,
                                    model,
                                    target_language
                                )
    
                            else:
                                result = review_azure(
                                    prompt,
                                    endpoint,
                                    api_key,
                                    model,
                                    target_language
                                )
    
                            apply_ai_result(record, result)
                            reviewed += 1
    
                        except Exception as exc:
                            record.ai_status = "ERROR"
                            record.ai_explanation = str(exc)
                            record.ai_model = model
                            record.ai_language = target_language
                            errors += 1
    
                        progress.progress(
                            idx / len(selected_for_ai)
                        )
    
                    st.session_state.stats = build_stats(
                        st.session_state.records
                    )
    
                    log(
                        f"AI review complete | provider={provider} | "
                        f"reviewed={reviewed} | errors={errors}"
                    )
    
                    st.success(
                        f"AI review complete. "
                        f"Reviewed {reviewed}. Errors: {errors}."
                    )
    
            st.divider()
            st.subheader("AI Review Results")
    
            df = records_to_dataframe(
                st.session_state.records
            )
    
            if "AI Status" in df.columns:
                ai_df = df[
                    df["AI Status"]
                    .fillna("Not reviewed")
                    != "Not reviewed"
                ]
            else:
                ai_df = pd.DataFrame()
    
            st.dataframe(
                ai_df,
                use_container_width=True,
                hide_index=True,
                height=420,
            )
    # ======================================================
    # TAB 6 MERGE CENTER
    # ======================================================

    with tabs[5]:
        if not st.session_state.records:
            st.info("Load files first.")
        else:
            st.subheader("Merge Center")

            st.write("Merge all loaded files into one clean export.")

            dedupe_mode = st.selectbox(
                "Deduplication",
                [
                    "No Deduplication",
                    "Source + Target",
                    "Source Only",
                ],
            )

            export_type = st.selectbox(
                "Merged Export Format",
                [
                    "tmx",
                    "xlsx",
                    "csv",
                    "xliff",
                ],
            )

            merged = dedupe_records(
                st.session_state.records,
                dedupe_mode,
            )

            st.info(
                f"Loaded records: {len(st.session_state.records)} | "
                f"After merge rules: {len(merged)}"
            )

            data, name, mime = write_by_type(
                merged,
                export_type,
            )

            st.download_button(
                f"Download Merged {export_type.upper()}",
                data=data,
                file_name=f"merged_{name}",
                mime=mime,
                use_container_width=True,
            )

    # ======================================================
    # TAB 7 EXPORT
    # ======================================================

    with tabs[6]:
        if not st.session_state.records:
            st.info("Nothing to export.")
        else:
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

            st.divider()

            grouped = defaultdict(list)

            for r in st.session_state.records:
                grouped[r.file_type].append(r)

            for file_type, group in grouped.items():
                data, name, mime = write_by_type(
                    group,
                    file_type,
                )

                st.download_button(
                    f"Download {file_type.upper()} Export",
                    data=data,
                    file_name=name,
                    mime=mime,
                    use_container_width=True,
                )

    # ======================================================
    # TAB 8 LOGS
    # ======================================================

    with tabs[7]:
        st.text_area(
            "Logs",
            "\n".join(st.session_state.logs),
            height=720,
        )


if __name__ == "__main__":
    main()
