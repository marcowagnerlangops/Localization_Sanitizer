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
APP_VERSION = "4.0"
MAKER_LINE = "Made by LangOps Solutions"


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
    flag_german_micro_qa = st.sidebar.checkbox("German Micro QA", True)
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


def rerun_qa(settings):
    QAEngine.apply(
        st.session_state.records,
        settings,
        st.session_state.brand_rules,
        st.session_state.glossary_rules,
    )
    st.session_state.stats = build_stats(st.session_state.records)


def filter_dataframe(df):
    if df.empty:
        return df

    with st.expander("Filters", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            severity = st.selectbox("Severity", ["All", "Issues", "OK"])
        with c2:
            lqa_severity = st.selectbox("LQA Severity", ["All", "Critical", "Major", "Minor", "OK", "Unscored"])
        with c3:
            file_type = st.selectbox("File Type", ["All"] + sorted(df["Type"].dropna().unique().tolist()))

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
            ai_status_values = ["All"] + sorted(df["AI Status"].fillna("Not reviewed").unique().tolist()) if "AI Status" in df.columns else ["All"]
            ai_status = st.selectbox("AI Status", ai_status_values)

    out = df.copy()

    if severity != "All":
        out = out[out["Severity"] == severity]
    if lqa_severity != "All":
        out = out[out["LQA Severity"] == lqa_severity]
    if file_type != "All":
        out = out[out["Type"] == file_type]
    if category != "All":
        out = out[out["Issue Categories"].fillna("").str.contains(category, case=False, regex=False)]
    if "ai_status" in locals() and ai_status != "All" and "AI Status" in out.columns:
        out = out[out["AI Status"].fillna("Not reviewed") == ai_status]
    if search.strip():
        needle = search.lower()
        out = out[
            out["Source"].astype(str).str.lower().str.contains(needle, regex=False)
            | out["Target"].astype(str).str.lower().str.contains(needle, regex=False)
            | out["Issue Details"].astype(str).str.lower().str.contains(needle, regex=False)
            | out["LQA Details"].astype(str).str.lower().str.contains(needle, regex=False)
            | out.get("AI Suggestion", "").astype(str).str.lower().str.contains(needle, regex=False)
            | out.get("AI Explanation", "").astype(str).str.lower().str.contains(needle, regex=False)
        ]
    return out


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
            key = (r.source_lang, r.target_lang, r.source_text.strip(), r.target_text.strip())
        elif mode == "Source Only":
            key = (r.source_lang, r.source_text.strip())
        else:
            key = r.record_id
        if key not in seen:
            seen.add(key)
            result.append(r)
    return result


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🧼", layout="wide")
    init_state()

    st.title("🧼 LangOps Sanitizer Pro")
    st.caption("Clean, QA, merge and export TMX, XLSX, CSV, XLIFF, XLF, TXLF and XLZ localization assets.")

    settings = sidebar_settings()

    st.sidebar.divider()
    st.sidebar.subheader("Brand Protection")
    brand_file = st.sidebar.file_uploader("Upload Brand Rules XLSX / CSV", type=["xlsx", "csv"], key="brand")
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
    glossary_file = st.sidebar.file_uploader("Upload Glossary XLSX / CSV", type=["xlsx", "csv"], key="glossary")
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

    tabs = st.tabs(["Upload & Analyze", "Dashboard", "Segments", "AI Review", "Merge Center", "Export", "Logs"])

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
                            st.session_state.records = parse_uploaded_files(uploaded, src_lang, tgt_lang)
                            changed = RepairEngine.apply(st.session_state.records, settings)
                            rerun_qa(settings)
                            log(f"Analysis complete | {len(st.session_state.records)} segments | {changed} repaired")
                        st.success("Analysis complete.")
                    except Exception as exc:
                        st.error(str(exc))
        with c2:
            if st.button("Run Auto Repairs", use_container_width=True):
                changed = RepairEngine.apply(st.session_state.records, settings)
                rerun_qa(settings)
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
            c8.metric("Glossary Terms", len(st.session_state.glossary_rules.rules))

            c9, c10 = st.columns(2)
            c9.metric("AI Reviewed", stats.get("ai_reviewed_segments", 0))
            ai_pass = stats.get("ai_status_counts", {}).get("PASS", 0)
            ai_review = stats.get("ai_status_counts", {}).get("REVIEW", 0)
            c10.metric("AI PASS / REVIEW", f"{ai_pass} / {ai_review}")

            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                if stats.get("issue_categories"):
                    st.subheader("Issue Categories")
                    chart = pd.DataFrame(stats["issue_categories"].items(), columns=["Category", "Count"])
                    st.bar_chart(chart, x="Category", y="Count")
            with col2:
                if stats.get("lqa_segment_severity"):
                    st.subheader("LQA Segment Severity")
                    sev_chart = pd.DataFrame(stats["lqa_segment_severity"].items(), columns=["Severity", "Count"])
                    st.bar_chart(sev_chart, x="Severity", y="Count")

            st.subheader("LQA Scoring Model")
            st.info(
                "Quality Score = 100 - Critical×10 - Major×5 - Minor×1. "
                "Critical includes missing targets, placeholder mismatches, number mismatches and malformed tags. "
                "Major includes glossary, brand, source=target and severe length-ratio issues. "
                "Minor includes typography, punctuation and German micro-QA issues."
            )

    with tabs[2]:
        if not st.session_state.records:
            st.info("No records loaded.")
        else:
            df = records_to_dataframe(st.session_state.records)
            filtered = filter_dataframe(df)
            st.dataframe(filtered, use_container_width=True, hide_index=True, height=720)


    with tabs[3]:
        if not st.session_state.records:
            st.info("Load and analyze files first.")
        else:
            st.subheader("AI Language Review")
            st.write(
                "Optional AI layer for grammar, fluency, idiomatic usage, and rewrite suggestions. "
                "Your API key is used only for this session and is not stored by the app."
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                api_key = st.text_input("OpenAI API Key", type="password", help="Use your own OpenAI API key.")
                model = st.text_input("Model", value="gpt-5.2")
            with c2:
                language = st.selectbox("Target Language for AI Review", LANGUAGE_OPTIONS, index=0)
                custom_language = ""
                if language == "Custom":
                    custom_language = st.text_input("Custom Target Language")
            with c3:
                strictness = st.selectbox("Review Strictness", STRICTNESS_OPTIONS, index=1)
                mode = st.selectbox("Review Scope", AI_REVIEW_MODES, index=0)
                max_segments = st.number_input("Max Segments per Run", min_value=1, max_value=500, value=25, step=5)

            custom_instructions = st.text_area(
                "Optional Custom AI Instructions",
                value="",
                placeholder="Example: Use a professional but friendly marketing tone. Keep product names unchanged.",
                height=90,
            )

            target_language = custom_language.strip() if language == "Custom" and custom_language.strip() else language
            selected_for_ai = select_records_for_ai(st.session_state.records, mode, int(max_segments))
            st.info(f"Segments selected for AI review: {len(selected_for_ai)}")

            if st.button("Run AI Review", type="primary", use_container_width=True):
                if not api_key.strip():
                    st.warning("Please enter your OpenAI API key.")
                elif not selected_for_ai:
                    st.warning("No segments match the selected AI review scope.")
                else:
                    progress = st.progress(0)
                    reviewed = 0
                    errors = 0
                    for idx, record in enumerate(selected_for_ai, start=1):
                        try:
                            result = review_segment_with_openai(
                                record=record,
                                api_key=api_key.strip(),
                                model=model.strip() or "gpt-5.2",
                                target_language=target_language,
                                strictness=strictness,
                                custom_instructions=custom_instructions,
                            )
                            apply_ai_result(record, result)
                            reviewed += 1
                        except Exception as exc:
                            record.ai_status = "ERROR"
                            record.ai_severity = ""
                            record.ai_suggestion = ""
                            record.ai_explanation = str(exc)
                            record.ai_model = model.strip() or "gpt-5.2"
                            record.ai_language = target_language
                            errors += 1
                        progress.progress(idx / len(selected_for_ai))
                    st.session_state.stats = build_stats(st.session_state.records)
                    log(f"AI review complete | reviewed={reviewed} | errors={errors} | model={model}")
                    st.success(f"AI review complete. Reviewed {reviewed} segment(s). Errors: {errors}.")

            st.divider()
            st.subheader("AI Review Results")
            df = records_to_dataframe(st.session_state.records)
            ai_df = df[df["AI Status"].fillna("Not reviewed") != "Not reviewed"]
            st.dataframe(ai_df, use_container_width=True, hide_index=True, height=420)

            st.subheader("Accept AI Suggestion")
            suggestion_records = [r for r in st.session_state.records if r.ai_suggestion]
            if not suggestion_records:
                st.caption("No AI suggestions available yet.")
            else:
                selected_id = st.selectbox("Select Record ID with AI Suggestion", [r.record_id for r in suggestion_records])
                selected_record = next(r for r in suggestion_records if r.record_id == selected_id)
                st.text_area("Current Target", value=selected_record.target_text, height=100, disabled=True)
                st.text_area("AI Suggestion", value=selected_record.ai_suggestion, height=100, disabled=True)
                if st.button("Accept Suggestion for Selected Record", use_container_width=True):
                    selected_record.target_text = selected_record.ai_suggestion
                    selected_record.notes = (selected_record.notes + "; " if selected_record.notes else "") + "AI suggestion accepted"
                    rerun_qa(settings)
                    log(f"Accepted AI suggestion for record {selected_id}")
                    st.success("AI suggestion accepted and rule-based QA re-run.")
                    st.rerun()

    with tabs[4]:
        if not st.session_state.records:
            st.info("Load files first.")
        else:
            st.subheader("Merge Center")
            st.write("Merge all loaded files into one clean export.")
            dedupe_mode = st.selectbox("Deduplication", ["No Deduplication", "Source + Target", "Source Only"])
            export_type = st.selectbox("Merged Export Format", ["tmx", "xlsx", "csv", "xliff"])
            merged = dedupe_records(st.session_state.records, dedupe_mode)
            st.info(f"Loaded records: {len(st.session_state.records)} | After merge rules: {len(merged)}")
            data, name, mime = write_by_type(merged, export_type)
            st.download_button(
                f"Download Merged {export_type.upper()}",
                data=data,
                file_name=f"merged_{name}",
                mime=mime,
                use_container_width=True,
            )

    with tabs[5]:
        if not st.session_state.records:
            st.info("Nothing to export.")
        else:
            report = build_xlsx_report(st.session_state.records, st.session_state.stats)
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
                data, name, mime = write_by_type(group, file_type)
                st.download_button(
                    f"Download {file_type.upper()} Export",
                    data=data,
                    file_name=name,
                    mime=mime,
                    use_container_width=True,
                )

    with tabs[6]:
        st.text_area("Logs", "\n".join(st.session_state.logs), height=720)


if __name__ == "__main__":
    main()
