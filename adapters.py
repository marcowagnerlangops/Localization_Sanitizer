from __future__ import annotations

import os
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET

import pandas as pd

from sanitizer_core import SegmentRecord


# ============================================================
# Shared helpers
# ============================================================

XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"


def _strip_namespace(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def _clean_text(text: str) -> str:
    import re
    return re.sub(r"\s+", " ", text or "").strip()


def _xml_lang(elem: ET.Element) -> str:
    return elem.attrib.get(XML_LANG) or elem.attrib.get("lang") or ""


def _find_source_target_columns(df: pd.DataFrame) -> Tuple[int, int]:
    cols = [str(c).strip().lower() for c in df.columns]

    src_candidates = {
        "source", "src", "english", "en",
        "source text", "source_text"
    }

    tgt_candidates = {
        "target", "trg", "translation",
        "german", "de",
        "target text", "target_text"
    }

    src = None
    tgt = None

    for i, col in enumerate(cols):
        if src is None and col in src_candidates:
            src = i
        if tgt is None and col in tgt_candidates:
            tgt = i

    if src is None or tgt is None:
        if len(df.columns) < 2:
            raise ValueError("At least two columns required.")
        return 0, 1

    return src, tgt


# ============================================================
# XLSX
# ============================================================

def read_xlsx(uploaded_file, start_id, source_lang, target_lang):
    df = pd.read_excel(uploaded_file)

    src_col, tgt_col = _find_source_target_columns(df)

    records = []

    for idx, row in df.iterrows():
        source = "" if pd.isna(row.iloc[src_col]) else str(row.iloc[src_col])
        target = "" if pd.isna(row.iloc[tgt_col]) else str(row.iloc[tgt_col])

        records.append(
            SegmentRecord(
                record_id=start_id + len(records),
                file_name=uploaded_file.name,
                file_type="xlsx",
                unit_id=str(idx + 1),
                source_lang=source_lang,
                target_lang=target_lang,
                source_text=source,
                target_text=target,
            )
        )

    meta = {"kind": "xlsx"}
    return records, meta


def write_xlsx(records):
    df = pd.DataFrame({
        "source": [r.source_text for r in records],
        "target": [r.target_text for r in records],
        "issue_status": [r.severity for r in records],
        "issue_categories": [r.issue_categories for r in records],
        "issue_details": [r.issue_details for r in records],
        "repair_actions": [r.repair_actions for r in records],
    })

    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)

    output.seek(0)
    return output.getvalue()


# ============================================================
# CSV
# ============================================================

def read_csv(uploaded_file, start_id, source_lang, target_lang):
    content = uploaded_file.getvalue()

    df = None
    last_error = None

    for enc in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
        for sep in [None, ",", ";", "\t", "|"]:
            try:
                buffer = BytesIO(content)

                if sep is None:
                    df = pd.read_csv(buffer, encoding=enc, sep=None, engine="python")
                else:
                    df = pd.read_csv(buffer, encoding=enc, sep=sep)

                break

            except Exception as exc:
                last_error = exc

        if df is not None:
            break

    if df is None:
        raise ValueError(f"Could not read CSV: {last_error}")

    src_col, tgt_col = _find_source_target_columns(df)

    records = []

    for idx, row in df.iterrows():
        source = "" if pd.isna(row.iloc[src_col]) else str(row.iloc[src_col])
        target = "" if pd.isna(row.iloc[tgt_col]) else str(row.iloc[tgt_col])

        records.append(
            SegmentRecord(
                record_id=start_id + len(records),
                file_name=uploaded_file.name,
                file_type="csv",
                unit_id=str(idx + 1),
                source_lang=source_lang,
                target_lang=target_lang,
                source_text=source,
                target_text=target,
            )
        )

    meta = {"kind": "csv"}
    return records, meta


def write_csv(records):
    df = pd.DataFrame({
        "source": [r.source_text for r in records],
        "target": [r.target_text for r in records],
        "issue_status": [r.severity for r in records],
        "issue_categories": [r.issue_categories for r in records],
        "issue_details": [r.issue_details for r in records],
        "repair_actions": [r.repair_actions for r in records],
    })

    return df.to_csv(index=False).encode("utf-8-sig")


# ============================================================
# TMX
# ============================================================

def read_tmx(uploaded_file, start_id):
    tree = ET.parse(uploaded_file)
    root = tree.getroot()

    records = []

    tus = root.findall(".//tu")

    for idx, tu in enumerate(tus, start=1):
        tuid = tu.attrib.get("tuid", str(idx))

        tuvs = tu.findall("./tuv")

        if len(tuvs) < 2:
            continue

        source_tuv = tuvs[0]
        target_tuv = tuvs[1]

        source_lang = _xml_lang(source_tuv)
        target_lang = _xml_lang(target_tuv)

        source_seg = source_tuv.find("./seg")
        target_seg = target_tuv.find("./seg")

        source_text = _clean_text("".join(source_seg.itertext()) if source_seg is not None else "")
        target_text = _clean_text("".join(target_seg.itertext()) if target_seg is not None else "")

        records.append(
            SegmentRecord(
                record_id=start_id + len(records),
                file_name=uploaded_file.name,
                file_type="tmx",
                unit_id=tuid,
                source_lang=source_lang,
                target_lang=target_lang,
                source_text=source_text,
                target_text=target_text,
            )
        )

    meta = {"kind": "tmx"}
    return records, meta


def write_tmx(records):
    import xml.etree.ElementTree as ET
    from xml.dom import minidom
    from io import BytesIO
    from datetime import datetime, timezone

    creationdate = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    root = ET.Element("tmx", version="1.4")

    ET.SubElement(
        root,
        "header",
        {
            "creationtool": "LangOps Converter",
            "creationtoolversion": "1.0",
            "segtype": "block",
            "o-tmf": "OTC",
            "adminlang": "en-US",
            "srclang": records[0].source_lang if records else "en-US",
            "datatype": "unknown",
            "creationdate": creationdate
        }
    )

    body = ET.SubElement(root, "body")

    for idx, r in enumerate(records, start=1):

        tu = ET.SubElement(body, "tu")

        ET.SubElement(
            tu,
            "prop",
            {"type": "Txt::Domain"}
        ).text = r.meta.get("domain", "sales_central")

        ET.SubElement(
            tu,
            "prop",
            {"type": "Txt::Product"}
        ).text = r.meta.get("product", "sales_transcreation")

        ET.SubElement(
            tu,
            "prop",
            {"type": "Txt::Origin"}
        ).text = r.meta.get(
            "origin",
            f"imported_from/{r.file_name}/row{idx}.spl"
        )

        tuv1 = ET.SubElement(
            tu,
            "tuv",
            {"xml:lang": r.source_lang or "en-US"}
        )
        seg1 = ET.SubElement(tuv1, "seg")
        seg1.text = r.source_text

        tuv2 = ET.SubElement(
            tu,
            "tuv",
            {"xml:lang": r.target_lang or "de-DE"}
        )
        seg2 = ET.SubElement(tuv2, "seg")
        seg2.text = r.target_text

    xml_bytes = ET.tostring(root, encoding="utf-8")
    pretty = minidom.parseString(xml_bytes).toprettyxml(
        indent="    ",
        encoding="utf-8"
    )

    return pretty


# ============================================================
# XLIFF / XLF / TXLF
# ============================================================

def read_xliff(uploaded_file, start_id):
    root = ET.fromstring(uploaded_file.getvalue())

    records = []

    units = [x for x in root.iter() if _strip_namespace(x.tag) in {"trans-unit", "unit"}]

    for idx, unit in enumerate(units, start=1):
        unit_id = unit.attrib.get("id", str(idx))

        source = ""
        target = ""

        for child in unit.iter():
            tag = _strip_namespace(child.tag)

            if tag == "source" and not source:
                source = _clean_text("".join(child.itertext()))

            if tag == "target" and not target:
                target = _clean_text("".join(child.itertext()))

        if source or target:
            records.append(
                SegmentRecord(
                    record_id=start_id + len(records),
                    file_name=uploaded_file.name,
                    file_type="xliff",
                    unit_id=unit_id,
                    source_lang="",
                    target_lang="",
                    source_text=source,
                    target_text=target,
                )
            )

    meta = {"kind": "xliff"}
    return records, meta


def write_xliff(records):
    root = ET.Element("xliff", version="1.2")

    file_elem = ET.SubElement(root, "file", {
        "source-language": records[0].source_lang if records else "",
        "target-language": records[0].target_lang if records else "",
        "datatype": "plaintext",
        "original": "sanitized"
    })

    body = ET.SubElement(file_elem, "body")

    for idx, r in enumerate(records, start=1):
        tu = ET.SubElement(body, "trans-unit", {"id": r.unit_id or str(idx)})

        src = ET.SubElement(tu, "source")
        src.text = r.source_text

        tgt = ET.SubElement(tu, "target")
        tgt.text = r.target_text

    output = BytesIO()
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)

    return output.getvalue()


# ============================================================
# XLZ
# ============================================================

def read_xlz(uploaded_file, start_id):
    records = []

    with zipfile.ZipFile(BytesIO(uploaded_file.getvalue()), "r") as zf:
        for name in zf.namelist():
            if name.lower().endswith((".xlf", ".xliff", ".txlf")):
                content = zf.read(name)

                dummy = BytesIO(content)
                dummy.name = name

                inner_records, _ = read_xliff(dummy, start_id + len(records))

                for r in inner_records:
                    r.file_type = "xlz"
                    r.meta["inner_file"] = name

                records.extend(inner_records)

    meta = {"kind": "xlz"}
    return records, meta


def write_xlz(records):
    output = BytesIO()

    grouped = {}

    for r in records:
        name = r.meta.get("inner_file", "sanitized.xliff")
        grouped.setdefault(name, []).append(r)

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, group in grouped.items():
            zf.writestr(name, write_xliff(group))

    return output.getvalue()


# ============================================================
# MAIN ROUTER
# ============================================================

def read_any(uploaded_file, start_id, source_lang="", target_lang=""):
    ext = Path(uploaded_file.name).suffix.lower()

    if ext == ".xlsx":
        return read_xlsx(uploaded_file, start_id, source_lang, target_lang)

    if ext == ".csv":
        return read_csv(uploaded_file, start_id, source_lang, target_lang)

    if ext == ".tmx":
        return read_tmx(uploaded_file, start_id)

    if ext in {".xlf", ".xliff", ".txlf"}:
        return read_xliff(uploaded_file, start_id)

    if ext == ".xlz":
        return read_xlz(uploaded_file, start_id)

    raise ValueError(f"Unsupported file type: {uploaded_file.name}")


def write_by_type(records, file_type):
    if file_type == "xlsx":
        return (
            write_xlsx(records),
            "sanitized_export.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    if file_type == "csv":
        return (
            write_csv(records),
            "sanitized_export.csv",
            "text/csv"
        )

    if file_type == "tmx":
        return (
            write_tmx(records),
            "sanitized_export.tmx",
            "application/xml"
        )

    if file_type in {"xliff", "xlf", "txlf"}:
        return (
            write_xliff(records),
            "sanitized_export.xliff",
            "application/xml"
        )

    if file_type == "xlz":
        return (
            write_xlz(records),
            "sanitized_export.xlz",
            "application/zip"
        )

    return (
        write_xlsx(records),
        "sanitized_export.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
