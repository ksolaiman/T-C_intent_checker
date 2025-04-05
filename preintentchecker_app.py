import streamlit as st
import pandas as pd
from pathlib import Path
from Clause_Extraction_Pipeline_v4_update_step4 import process_documents, normalize_intent, detect_violation, load_local_document

st.set_page_config(page_title="Pre-Intent Checker", layout="wide")
st.title("🔍 Pre-Intention Violation Checker")

uploaded_files = st.file_uploader(
    "Upload T&C Documents (PDF or DOCX)",
    accept_multiple_files=True,
    type=["pdf", "docx"]
)

intent = st.text_input("What do you want to do? (e.g., Buy stock with unsettled funds)")

if uploaded_files and intent:
    raw_docs = {}
    for f in uploaded_files:
        file_path = Path(f.name)
        file_path.write_bytes(f.read())
        text = load_local_document(str(file_path))
        raw_docs[f.name] = text

    st.info("Processing clauses...")
    clause_df = process_documents(raw_docs)

    st.success(f"Extracted {len(clause_df)} clauses.")
    violation = detect_violation(normalize_intent(intent), clause_df)

    if violation['violation']:
        st.error("⚠️ Potential Violation Detected")
        st.write(violation['explanation'])
        st.code(violation['matched_clause'], language='text')
        st.write("**Tags:**", violation['tags'])
    else:
        st.success("✅ No matching violation found.")

    with st.expander("See All Processed Clauses"):
        st.dataframe(clause_df[['clause', 'quality_score', 'manual_tags', 'referential', 'bullet']])
else:
    st.warning("Please upload documents and enter your intended action.")
