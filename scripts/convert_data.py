import os
import re
import pandas as pd
from transformers import pipeline

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Initialize medical NER pipeline for fallback extraction
ner = pipeline(
    "ner",
    model="iioSnail/bert-base-chinese-medical-ner",
    aggregation_strategy="simple"
)

# Load all Excel templates into DataFrames
template_dfs = {}
for fn in os.listdir(TEMPLATES_DIR):
    if fn.endswith(".xlsx"):
        tpl_name = os.path.splitext(fn)[0]
        path = os.path.join(TEMPLATES_DIR, fn)
        # Read as strings to preserve leading zeros etc.
        df = pd.read_excel(path, dtype=str)
        template_dfs[tpl_name] = df


def extract_subject_id(text: str) -> str:
    """
    Extract the subject ID from the OCR text via regex or NER fallback.
    """
    # 1) Regex for explicit labels
    m = re.search(r"(受试者编号|Subject\s*ID)\s*[:：]?\s*(\w+)", text)
    if m:
        return m.group(2)
    # 2) NER-based fallback
    ents = ner(text)
    for ent in ents:
        if ent.get("entity_group") == "ID" and len(ent.get("word", "")) >= 3:
            return ent["word"]
    return ""


def extract_field_value(text: str, field_name: str, keywords: list[str] = None) -> str:
    """
    Extract a field's value by regex matching on its name or provided keywords,
    and fallback to NER if needed.
    """
    # Regex on the field name
    pattern = rf"{re.escape(field_name)}\s*[:：]?\s*(\S+)"
    m = re.search(pattern, text)
    if m:
        return m.group(1)
    # Try additional keywords
    if keywords:
        for kw in keywords:
            m2 = re.search(rf"{re.escape(kw)}\s*[:：]?\s*(\S+)", text)
            if m2:
                return m2.group(1)
    # NER fallback: match if field_name appears in an entity
    ents = ner(text)
    for ent in ents:
        if field_name in ent.get("word", ""):
            return ent["word"]
    return ""


def map_to_templates(text_blob: str) -> dict[str, pd.DataFrame]:
    """
    Map extracted text into each Excel template DataFrame.
    Returns a dict of filled DataFrames keyed by template name.
    """
    subject_id = extract_subject_id(text_blob)
    filled = {}

    for tpl_name, df in template_dfs.items():
        df_copy = df.copy()
        # Write subject_id into any matching column
        for col in df_copy.columns:
            if col.lower() in ("subject_id", "受试者编号"):
                df_copy.at[0, col] = subject_id
        # Fill other fields
        for col in df_copy.columns:
            col_lower = col.lower()
            if col_lower in ("subject_id",):
                continue
            # Optionally, provide synonyms or keywords from a separate config
            val = extract_field_value(text_blob, col)
            df_copy.at[0, col] = val
        filled[tpl_name] = df_copy

    return filled
