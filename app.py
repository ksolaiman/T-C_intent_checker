import streamlit as st
from sentence_transformers import SentenceTransformer, util

# Load SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Clause database
clauses = [
    {
        "id": "BILT001",
        "text": "The Bilt Mastercard may not be used for cash-like transactions such as money orders, wire transfers, or similar instruments.",
        "domain": "credit",
        "category": "prohibited",
        "risk": "medium",
        "source": "Bilt T&C"
    },
    {
        "id": "FID001",
        "text": "Unsettled proceeds from the sale of securities cannot be used to purchase new securities until the T+2 settlement period is complete.",
        "domain": "brokerage",
        "category": "restricted",
        "risk": "high",
        "source": "Fidelity Brokerage T&C"
    }
]

# Streamlit app
st.title("Intent-Based T&C Compliance Assistant")

st.markdown("Type your intended action below. The assistant will warn you if it may violate any known Terms & Conditions.")

user_input = st.text_area("🔍 Describe your intended action:", height=100)

if user_input:
    with st.spinner("Checking against T&C clauses..."):
        intent_embedding = model.encode(user_input, convert_to_tensor=True)
        clause_embeddings = [model.encode(c["text"], convert_to_tensor=True) for c in clauses]
        scores = [util.pytorch_cos_sim(intent_embedding, emb)[0][0].item() for emb in clause_embeddings]

        best_match_idx = scores.index(max(scores))
        best_score = scores[best_match_idx]

        threshold = 0.5  # similarity threshold
        if best_score > threshold:
            matched_clause = clauses[best_match_idx]
            st.error(f"⚠️ Potential Violation Detected (Score: {best_score:.2f})")
            st.markdown(f"**Matched Clause:** {matched_clause['text']}")
            st.markdown(f"**Domain:** {matched_clause['domain'].capitalize()}")
            st.markdown(f"**Risk Level:** {matched_clause['risk'].capitalize()}")
            st.markdown(f"**Source:** {matched_clause['source']}")
        else:
            st.success("✅ No violation detected based on known clauses.")

st.markdown("---")
st.caption("Prototype using SBERT for semantic similarity. Clause database includes Bilt and Fidelity examples.")
