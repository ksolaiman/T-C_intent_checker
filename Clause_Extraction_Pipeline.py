# Clause Extraction Pipeline - Step-by-Step Modules

import re
import nltk
import spacy
import pandas as pd
from typing import List, Dict
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import docx
import os

nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Step 1: Clause Corpus Collection (Automated from Online Sources or Manual Upload) ---

def fetch_online_tos(name: str) -> str:
    """Fetch raw text of Terms from a known URL for a brokerage or credit card."""
    urls = {
        "fidelity": "https://www.fidelity.com/bin-public/060_www_fidelity_com/documents/customer-service/customer-agreement.pdf",
        "bilt": "https://www.wellsfargo.com/credit-cards/agreements/bilt-agreement/",
        "schwab": "https://www.schwab.com/legal/schwab-one-account-agreement",
        "amex": "https://www.americanexpress.com/en-us/legal/cardmember-agreements/",
    }
    if name not in urls:
        raise ValueError("Unsupported provider")
    response = requests.get(urls[name])
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text()

def load_local_document(file_path: str) -> str:
    """Extract text from local PDF or DOCX document."""
    if file_path.endswith(".pdf"):
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        return "".join([para.text for para in doc.paragraphs if para.text.strip()])
    else:
        raise ValueError("Unsupported file format. Only PDF and DOCX are allowed.")

# --- Step 2: Clause Segmentation and Preprocessing ---

def segment_sentences(doc_text: str) -> List[str]:
    return nltk.sent_tokenize(doc_text)

def extract_candidate_clauses(sentences: List[str]) -> List[Dict]:
    """
    Extract clauses by grouping together dependent sentences.
    If a sentence starts with a referential cue (e.g., "This", "Such", "These"), 
    it is merged with the previous sentence.
    """
    clauses = []  # list of dicts with 'text', 'referential', 'bullet'
    buffer = ""
    for sentence in sentences:
        sent = sentence.strip()
        if not sent:
            continue
        is_referential = bool(re.match(r"^(This|These|Such|It|They|Those)", sent))
        is_bullet = bool(re.match(r"^[\-\*•]\s", sent)) or sent.endswith(":")

        if is_referential:
            buffer += " " + sent
        elif is_bullet:
            buffer += "" + sent
        else:
            if buffer:
                clauses.append({"text": buffer.strip(), "referential": is_referential, "bullet": is_bullet})
            buffer = sent
    if buffer:
        clauses.append({"text": buffer.strip(), "referential": False, "bullet": False})
    return clauses

# --- Step 3: Clause Typing and Tag Suggestion ---

LEGAL_KEYWORDS = ["may not", "must", "require", "subject to", "prohibited", "not permitted", "unless"]
DOMAIN_KEYWORDS = ["settlement", "donation", "crypto", "margin", "unsettled", "gambling", "recurring"]

TYPES = ["prohibited", "conditionally allowed", "allowed", "ambiguous"]

def suggest_tags(clause: str) -> List[str]:
    tags = [k for k in DOMAIN_KEYWORDS if k in clause.lower()]
    return tags

def get_embedding_tags(clause: str, tag_pool: List[str] = DOMAIN_KEYWORDS) -> List[str]:
    clause_emb = model.encode(clause, convert_to_tensor=True)
    tag_embs = model.encode(tag_pool, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(clause_emb, tag_embs)[0]
    return [tag_pool[i] for i, score in enumerate(scores) if score.item() > 0.5]

# --- Clause Scoring Heuristics ---

def score_clause(entry: Dict) -> Dict:
    clause = entry['text']
    words = clause.split()
    length_score = max(0, min(1, len(words)/100)) if len(words) >= 5 else 0
    keyword_score = 1.0 if any(k in clause.lower() for k in LEGAL_KEYWORDS) else 0.0
    punct_score = 1.0 if clause.strip().endswith(".") else 0.5
    manual_tags = suggest_tags(clause)
    embed_tags = get_embedding_tags(clause)
    tag_score = 1.0 if manual_tags or embed_tags else 0.0
    struct_score = 0.8  # Placeholder structural score
    total_score = round((length_score + keyword_score + punct_score + tag_score + struct_score) / 5, 2)
    review_flag = total_score < 0.7

    return {
        "clause": clause,
        "referential": entry.get('referential', False),
        "bullet": entry.get('bullet', False),
        "length_score": round(length_score, 2),
        "keyword_score": keyword_score,
        "punctuation_score": punct_score,
        "domain_score": tag_score,
        "structure_score": struct_score,
        "quality_score": total_score,
        "review_needed": review_flag,
        "manual_tags": manual_tags,
        "embed_tags": embed_tags
    }

# --- Step 4: Intent Input Processing (Mocked) ---

def normalize_intent(intent: str) -> str:
    return intent.strip().lower()

# --- Step 5: Semantic Matching ---

def semantic_match(intent: str, clauses: List[str]) -> List[Dict]:
    intent_vec = model.encode(intent, convert_to_tensor=True)
    clause_vecs = model.encode(clauses, convert_to_tensor=True)
    sims = util.pytorch_cos_sim(intent_vec, clause_vecs)[0]
    return sorted([(clauses[i], sims[i].item()) for i in range(len(clauses))], key=lambda x: -x[1])

# --- Step 6: Violation Detection and Feedback ---

def detect_violation(intent: str, clause_db: pd.DataFrame, threshold: float = 0.5) -> Dict:
    clause_texts = clause_db['clause'].tolist()
    matches = semantic_match(intent, clause_texts)
    if matches and matches[0][1] > threshold:
        clause = matches[0][0]
        row = clause_db[clause_db['clause'] == clause].iloc[0]
        return {
            "violation": True,
            "matched_clause": clause,
            "score": matches[0][1],
            "tags": row.manual_tags + row.embed_tags,
            "explanation": f"Your action may conflict with the following clause: '{clause}'"
        }
    return {"violation": False}

# --- Pipeline Orchestration ---

def process_documents(raw_docs: Dict[str, str]) -> pd.DataFrame:
    all_clauses = []
    for source, text in raw_docs.items():
        sentences = segment_sentences(text)
        candidates = extract_candidate_clauses(sentences)
        for clause_entry in candidates:
            scored = score_clause(clause_entry)
            scored["source"] = source
            all_clauses.append(scored)
    return pd.DataFrame(all_clauses)

# --- Export to CSV for Manual Review ---

def export_review_csv(df: pd.DataFrame, path: str = "clauses_review.csv"):
    df.to_csv(path, index=False)

# --- Run the whole pipeline ---
if __name__ == "__main__":
    providers = ["fidelity", "bilt"]
    docs = {p: fetch_online_tos(p) for p in providers}
    df_clauses = process_documents(docs)
    export_review_csv(df_clauses)
    print(df_clauses.head())
