# ------------------ Part 1 of 3 ------------------
# Imports + Helper Utilities + Resume Parsing & Scoring Helpers
import os
import re
import io
import math
import logging
from collections import Counter
from typing import List, Tuple, Dict, Any, Optional

# optional NLP libs
try:
    import spacy
except Exception:
    spacy = None

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import docx2txt
except Exception:
    docx2txt = None

# embeddings & sentence transformers
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# Optional OpenAI integration for interactive chatbot (safe fallback if not available)
try:
    import openai
except Exception:
    openai = None

# load .env if present
from dotenv import load_dotenv
load_dotenv()

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------ BASE SKILLS (role -> core skills) ------------------
# Keep this small but extendable. Your UI's sidebar uses BASE_SKILLS keys.
BASE_SKILLS = {
    "AI Engineer": [
        "python", "pytorch", "tensorflow", "machine learning", "deep learning",
        "nlp", "natural language processing", "transformers", "llm", "runnable models",
        "docker", "fastapi", "mlops", "mlem", "faiss", "vector database", "sql", "pandas", "numpy"
    ],
    "Data Scientist": [
        "python", "pandas", "numpy", "scikit-learn", "sklearn", "statistics", "visualization",
        "sql", "feature engineering", "machine learning", "deep learning"
    ],
    "ML Engineer": [
        "python", "docker", "kubernetes", "ci/cd", "mlflow", "pytorch", "tensorflow", "mlops"
    ],
    # add other roles used by your UI
}

# ------------------ Text Extraction Helpers ------------------
def extract_text_from_pdf_stream(stream: io.BytesIO) -> str:
    """Extract text from bytes stream of a PDF using PyMuPDF or fallback to empty."""
    if fitz is None:
        # fallback: no PyMuPDF installed
        return ""
    try:
        doc = fitz.open(stream=stream.read(), filetype="pdf")
        texts = []
        for page in doc:
            try:
                txt = page.get_text("text")
            except Exception:
                txt = ""
            if txt:
                texts.append(txt)
        return "\n\n".join(texts)
    except Exception as e:
        logger.debug(f"PDF extraction failed: {e}")
        return ""

def extract_text_from_docx_stream(stream: io.BytesIO) -> str:
    """Extract text from a DOCX bytes stream using docx2txt (works on file path only) - fallback uses docx2txt by writing to temp file."""
    if docx2txt is None:
        return ""
    try:
        import tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
        tmp.write(stream.read())
        tmp.flush()
        tmp.close()
        text = docx2txt.process(tmp.name) or ""
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
        return text
    except Exception as e:
        logger.debug(f"DOCX extraction failed: {e}")
        return ""

def extract_text_from_file(file_obj) -> str:
    """
    Accepts a Streamlit uploaded file-like object or any file object and returns extracted text.
    Supports PDF and DOCX. If neither available, tries to read raw text.
    """
    if hasattr(file_obj, "getbuffer"):  # Streamlit UploadFile
        bio = io.BytesIO(file_obj.getbuffer())
    else:
        # assume file-like
        try:
            bio = io.BytesIO(file_obj.read())
        except Exception:
            return ""

    # Try PDF first
    text = extract_text_from_pdf_stream(io.BytesIO(bio.getvalue()))
    if text and text.strip():
        return text

    # Try DOCX
    text = extract_text_from_docx_stream(io.BytesIO(bio.getvalue()))
    if text and text.strip():
        return text

    # Fallback: try to decode as utf-8
    try:
        raw = bio.getvalue().decode("utf-8", errors="ignore")
        if len(raw.strip()) > 30:
            return raw
    except Exception:
        pass

    return ""

# ------------------ Basic Text Cleaning ------------------
def clean_text(txt: str) -> str:
    """Lightweight cleaning: normalize whitespace, fix common OCR artifacts, remove long numeric blocks."""
    if not txt:
        return ""
    s = txt
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"[-]{4,}", "-", s)
    s = re.sub(r"[=]{3,}", "=", s)
    s = s.strip()
    return s

# ------------------ Simple Section Parsers ------------------
_re_year = re.compile(r"(19|20)\d{2}")

def extract_education(text: str) -> List[str]:
    if not text:
        return []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    edu_lines = []
    keywords = ["b.tech", "b.e.", "b.e", "btech", "m.tech", "m.e", "m.tech", "bachelor", "master", "phd", "mba", "degree", "graduat"]
    for ln in lines:
        low = ln.lower()
        if any(k in low for k in keywords):
            snippet = ln
            edu_lines.append(snippet)
    seen = []
    for e in edu_lines:
        if e not in seen:
            seen.append(e)
    return seen[:5]

def extract_years_experience(text: str) -> int:
    if not text:
        return 0
    m = re.search(r"(\d+)\s+years?", text.lower())
    if m:
        try:
            return int(m.group(1))
        except:
            pass
    years = [int(y.group(0)) for y in _re_year.finditer(text)]
    if len(years) >= 2:
        return max(years) - min(years)
    return 0

# ------------------ Skills Extraction ------------------
def tokenize_for_skills(text: str) -> List[str]:
    tokens = re.split(r"[,\n/()|:;]+", text)
    cleaned = []
    for t in tokens:
        s = t.strip()
        if not s:
            continue
        if len(s.split()) > 6:
            continue
        cleaned.append(s)
    return cleaned

def extract_skills(text: str, role_skills: Optional[List[str]] = None) -> List[str]:
    """
    Much smarter skill extractor:
    - Normalizes text
    - Performs substring + fuzzy-ish matching
    - Handles sklearn vs scikit-learn
    - Handles ML / Machine Learning / deep learning variants
    - Dedupes everything
    """
    if not text:
        return []

    txt = text.lower()

    # Build unified skill pool
    pool = set()
    for role, skills in BASE_SKILLS.items():
        for s in skills:
            pool.add(s.lower().strip())

    if role_skills:
        for s in role_skills:
            pool.add(s.lower().strip())

    extras = [
        "python", "java", "c++", "c", "javascript", "react", "node",
        "docker", "kubernetes", "sql", "nosql", "mlops",
        "sklearn", "scikit learn", "scikitlearn", "machine learning", "deep learning", "feature engineering"
    ]
    for x in extras:
        pool.add(x.lower())

    # Skill Matching
    found = set()

    # direct substring matching
    for skill in pool:
        if skill in txt:
            found.add(skill)
        # fuzzy pairing for sklearn variants
        if "sklearn" in skill and ("scikit" in txt or "sci-kit" in txt):
            found.add("scikit-learn")
        if "scikit" in skill and "sklearn" in txt:
            found.add("scikit-learn")

    # tokenized fallback
    words = re.findall(r"[A-Za-z0-9\+\-\#\.]+", txt)
    for w in words:
        w = w.strip().lower()
        if w in pool:
            found.add(w)

    # Formatting
    nice = []
    for s in sorted(found):
        if s.upper() in {"ML", "AI", "NLP", "LLM", "SQL"}:
            nice.append(s.upper())
        else:
            nice.append(s.capitalize())

    return nice

# ------------------ Scoring & Semantic Similarity Helpers ------------------
def simple_semantic_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    ta = set([w.lower() for w in re.findall(r"\w+", a) if len(w) > 2])
    tb = set([w.lower() for w in re.findall(r"\w+", b) if len(w) > 2])
    if not ta or not tb:
        return 0.0
    inter = ta.intersection(tb)
    score = len(inter) / math.sqrt(len(ta) * len(tb))
    return round(float(score), 3)

def calculate_skill_match_percent(resume_skills: List[str], jd_text: str, role_skills: Optional[List[str]] = None) -> Tuple[float, List[str]]:
    # normalize all skills
    resume_lower = {s.lower().strip() for s in resume_skills}

    # normalize required skill list
    if role_skills:
        required = {s.lower().strip() for s in role_skills}
    else:
        jd_tokens = [w.lower() for w in re.findall(r"[A-Za-z\+\-\#\#]+", jd_text)]
        freq = Counter(jd_tokens)
        required = {tok for tok, _ in freq.most_common(10)}

    # fuzzy/substring matching
    present = set()
    for req in required:
        for r in resume_lower:
            if req in r or r in req:
                present.add(req)

    missing = sorted(list(required - present))

    pct = round((len(present) / max(1, len(required))) * 100, 2)
    return pct, missing

def score_resume(text: str, jd_text: str, role_skills: Optional[List[str]] = None) -> Dict[str, Any]:
    skills = extract_skills(text, role_skills)
    skill_pct, missing = calculate_skill_match_percent(skills, jd_text, role_skills)
    semsim = simple_semantic_similarity(text, jd_text)
    semsim_pct = round(semsim * 100, 2)
    combined = round(0.6 * skill_pct + 0.4 * semsim_pct, 2)
    return {
        "skill_match_pct": skill_pct,
        "semantic_sim": semsim,
        "combined_score": combined,
        "missing_skills": missing
    }

# ------------------ Recommendation Generator ------------------
def generate_recommendation(text: str, jd_text: str, role_skills: Optional[List[str]] = None) -> Dict[str, Any]:
    skills = extract_skills(text, role_skills)
    skill_pct, missing = calculate_skill_match_percent(skills, jd_text, role_skills)

    summary_parts = []
    if skill_pct >= 75:
        summary_parts.append("Strong match on core skills.")
    elif skill_pct >= 40:
        summary_parts.append("Partial skill match — consider adding targeted skills.")
    else:
        summary_parts.append("Low skill overlap with JD — add required technologies and keywords.")

    if missing:
        summary_parts.append(f"Missing skills: {', '.join(missing[:6])}.")

    years = extract_years_experience(text)
    if years <= 1:
        summary_parts.append("Add project durations and measurable outcomes to showcase impact.")
    else:
        summary_parts.append(f"Listed {years} years of experience — highlight key achievements.")

    recommendations = [
        "Add specific project results with metrics (e.g., latency, accuracy, users).",
        "List toolchain used (PyTorch/TensorFlow, Docker, FastAPI).",
        "Highlight internships/projects with duration and measurable contributions.",
        "Use action verbs and quantify wherever possible."
    ]
    if missing:
        recommendations.insert(0, "Add or highlight these missing skills: " + ", ".join(missing[:6]))

    return {"summary": " ".join(summary_parts), "recommendations": recommendations}

# ------------------ Interactive Chatbot (OpenAI optional) ------------------
OPENAI_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API") or None
if openai and OPENAI_KEY:
    openai.api_key = OPENAI_KEY

def interactive_chatbot(user_q: str, resume_text: str = "", jd_text: str = "") -> str:
    system_prompt = (
        "You are a helpful hiring assistant. The user will ask about a candidate. "
        "Use the provided resume text and job description to produce HR-friendly advice. "
        "If information is missing, be explicit about it."
    )

    context = f"JOB DESCRIPTION:\n{jd_text}\n\nRESUME:\n{resume_text}"

    if openai and OPENAI_KEY:
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{user_q}\n\nContext:\n{context}"}
                ],
                temperature=0.0,
                max_tokens=300
            )
            answer = resp["choices"][0]["message"]["content"].strip()
            return answer
        except Exception as e:
            logger.warning(f"OpenAI call failed: {e}")

    user_lower = user_q.lower()
    if "skills" in user_lower or "missing" in user_lower:
        rec = generate_recommendation(resume_text, jd_text)
        return rec["summary"] + "\n\nRecommendations:\n- " + "\n- ".join(rec["recommendations"])
    if "highlight" in user_lower or "what to add" in user_lower or "improve" in user_lower:
        rec = generate_recommendation(resume_text, jd_text)
        return rec["summary"] + "\n\nTop suggestions:\n- " + "\n- ".join(rec["recommendations"][:3])
    return "I can help analyze skills, missing items, or suggest improvements. Try: 'What skills am I missing?' or 'How to improve this resume?'"
# ------------------ Part 1 End ------------------
# ------------------ Part 2 of 3 ------------------
# Embeddings, similarity helpers, and optional enhanced scoring

import threading
from functools import lru_cache

# Try to use sentence-transformers if available for better semantic similarity
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
_embed_lock = threading.Lock()
_embedding_model = None

def load_embedding_model():
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    if SentenceTransformer is None:
        return None
    with _embed_lock:
        if _embedding_model is None:
            try:
                _embedding_model = SentenceTransformer(EMBED_MODEL_NAME)
            except Exception as e:
                logger.warning(f"Failed to load embedding model '{EMBED_MODEL_NAME}': {e}")
                _embedding_model = None
    return _embedding_model

def embed_texts(texts: List[str]) -> Optional[List[List[float]]]:
    if not texts:
        return None
    model = load_embedding_model()
    if model is None:
        return None
    try:
        embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embs
    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
        return None

def cosine_sim(a, b):
    try:
        import numpy as _np
        a_arr = _np.array(a, dtype=float)
        b_arr = _np.array(b, dtype=float)
        if a_arr.size == 0 or b_arr.size == 0:
            return 0.0
        denom = (_np.linalg.norm(a_arr) * _np.linalg.norm(b_arr))
        if denom == 0:
            return 0.0
        return float(_np.dot(a_arr, b_arr) / denom)
    except Exception:
        try:
            dot = sum([float(x) * float(y) for x, y in zip(a, b)])
            norm_a = math.sqrt(sum([float(x) ** 2 for x in a]))
            norm_b = math.sqrt(sum([float(x) ** 2 for x in b]))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)
        except Exception:
            return 0.0

def calculate_similarity(text_a: str, text_b: str) -> float:
    if not text_a or not text_b:
        return 0.0
    model = load_embedding_model()
    if model is not None:
        try:
            embs = model.encode([text_a, text_b], convert_to_numpy=True, show_progress_bar=False)
            sim = cosine_sim(embs[0], embs[1])
            return round(float(sim), 3)
        except Exception as e:
            logger.debug(f"Embedding similarity failed, falling back: {e}")
            return simple_semantic_similarity(text_a, text_b)
    return simple_semantic_similarity(text_a, text_b)

def enhanced_score_resume(text: str, jd_text: str, role_skills: Optional[List[str]] = None) -> Dict[str, Any]:
    base = score_resume(text, jd_text, role_skills)
    emb_sim = calculate_similarity(text, jd_text)
    if emb_sim and emb_sim > 0:
        base["semantic_sim"] = round(float(emb_sim), 3)
        base["semantic_sim_pct"] = round(base["semantic_sim"] * 100, 2)
        sem_pct = base["semantic_sim"] * 100
        combined = round(0.6 * base["skill_match_pct"] + 0.4 * sem_pct, 2)
        base["combined_score"] = combined
    else:
        base["semantic_sim_pct"] = round(base["semantic_sim"] * 100, 2)
    return base

@lru_cache(maxsize=128)
def cached_embed_single(text: str):
    model = load_embedding_model()
    if model is None or not text:
        return None
    try:
        emb = model.encode([text], convert_to_numpy=True, show_progress_bar=False)
        return emb[0].tolist()
    except Exception:
        return None
# ------------------ Part 2 End ------------------
# ------------------ Part 3 of 3 ------------------
# Streamlit UI — Fully Upgraded HR-Friendly Dashboard

import streamlit as st
import pandas as pd
import time

st.set_page_config(
    page_title="AI Resume Screening Suite",
    layout="wide",
    page_icon="🧠",
)

# ---- Custom UI Theme ----
st.markdown("""
<style>
/* CARD STYLING */
.custom-card {
    padding: 20px;
    border-radius: 15px;
    background: #ffffff33;
    backdrop-filter: blur(8px);
    border: 1px solid #ffffff22;
    margin-bottom: 20px;
}

/* METRIC BOXES */
.metric-card {
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    font-weight: bold;
    background: linear-gradient(135deg, #0d47a1, #1976d2);
    color: white;
}

/* TITLES */
.big-title {
    font-size: 34px;
    font-weight: 800;
    color: #0d47a1;
}

/* ANIMATE FADE-IN */
@keyframes fadein {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0px); }
}

.fade-in { animation: fadein 0.8s ease-in-out; }

/* BUTTONS */
.stButton>button {
    background-color: #0d47a1 !important;
    color: white !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
}
</style>
""", unsafe_allow_html=True)

# ---- HEADER ----
st.markdown("<h1 class='big-title fade-in'>📄 AI-Powered Resume Screening Suite</h1>",
            unsafe_allow_html=True)
st.write("Smart. Fast. HR-grade insights — powered by NLP + Embeddings + Scoring Engine.")

st.divider()

# ----- Persistent session state initialization -----
if "selected_role" not in st.session_state:
    st.session_state.selected_role = "Data Scientist"
if "top_k" not in st.session_state:
    st.session_state.top_k = 5
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "jd_text" not in st.session_state:
    st.session_state.jd_text = ""
if "results" not in st.session_state:
    st.session_state.results = []
if "candidate_choice" not in st.session_state:
    st.session_state.candidate_choice = None
if "user_q" not in st.session_state:
    st.session_state.user_q = ""

# ---- SIDEBAR FILTERS ----
with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state.selected_role = st.selectbox(
        "Filter by role",
        ["AI Engineer", "Data Scientist", "ML Engineer"],
        index=["AI Engineer", "Data Scientist", "ML Engineer"].index(st.session_state.selected_role)
    )
    st.session_state.top_k = st.slider("Top K resumes to save", 1, 20, st.session_state.top_k)
    st.info(f"Screening for: **{st.session_state.selected_role}**\nMaximum shortlisted: **{st.session_state.top_k}**")

# ---- FILE UPLOAD ----
uploaded_files = st.file_uploader(
    "📑 Upload Resumes (PDF/DOCX) — multiple allowed",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

# persist uploads to session_state so reruns don't lose them
if uploaded_files:
    st.session_state.uploaded_files = uploaded_files
else:
    uploaded_files = st.session_state.uploaded_files

# ---- JOB DESCRIPTION INPUT ----
st.subheader("🧾 Job Description")
st.session_state.jd_text = st.text_area(
    "Paste the Job Description here",
    value=st.session_state.jd_text,
    height=180,
    placeholder="Enter JD..."
)
jd_text = st.session_state.jd_text

# ---- PROCESS BUTTON ----
process_btn = st.button("🚀 Run Screening")

# If results exist in session_state, reuse them. Only re-run when process_btn clicked.
if process_btn:
    # reset previous results
    st.session_state.results = []

    if not uploaded_files:
        st.error("Upload at least one resume.")
    elif not jd_text.strip():
        st.error("Paste a Job Description.")
    else:
        with st.spinner("Crunching data… extracting text… comparing skills… scoring candidates…"):
            time.sleep(0.8)
            for f in uploaded_files:
                try:
                    text = extract_text_from_file(f)
                    score = enhanced_score_resume(text, jd_text, None)
                    score["filename"] = getattr(f, "name", str(f))
                    score["text"] = text
                    st.session_state.results.append(score)
                except Exception as e:
                    st.error(f"Error reading {getattr(f, 'name', str(f))}: {e}")

        st.success("Screening complete! 🎉")

# Use results from session_state (persisted)
results = st.session_state.results or []

# only render if we have results
if results:
    df = pd.DataFrame(results).sort_values("combined_score", ascending=False).reset_index(drop=True)

    # store dataframe in session_state for navigation stability
    st.session_state._df = df

    # ---- CANDIDATE RANKING CHART ----
    st.markdown("## 📊 Candidate Rankings")
    try:
        st.bar_chart(df.set_index("filename")["combined_score"])
    except Exception:
        st.write("Unable to render chart for results.")

    # ---- TOP CHART ----
    st.markdown("## 🏆 Top Candidates")
    top_k = min(st.session_state.top_k, len(df))
    top_df = df.head(top_k)

    for _, row in top_df.iterrows():
        sem_pct = row.get("semantic_sim_pct", round(row.get("semantic_sim", 0) * 100, 2))
        st.markdown(f"""
        <div class="custom-card fade-in">
            <h3>👤 {row['filename']}</h3>
            <div class="metric-card">
                <p>Combined Score: {row['combined_score']}%</p>
            </div>

            <p><b>Skill Match:</b> {row['skill_match_pct']}%</p>
            <p><b>Semantic Similarity:</b> {sem_pct}%</p>
        </div>
        """, unsafe_allow_html=True)

    # ---- INSPECT SECTION ----
    st.divider()
    st.markdown("## 🎯 Inspect Candidate")

    filenames = df["filename"].tolist()
    if st.session_state.candidate_choice not in filenames:
        st.session_state.candidate_choice = filenames[0]

    st.session_state.candidate_choice = st.selectbox("Choose candidate", filenames, index=filenames.index(st.session_state.candidate_choice))
    selected = df[df["filename"] == st.session_state.candidate_choice].iloc[0]

    sem_pct_display = selected.get("semantic_sim_pct", round(selected.get("semantic_sim", 0) * 100, 2))

    st.markdown(f"""
    <div class="custom-card fade-in">
        <h3>📄 {selected['filename']}</h3>
        <p><b>Combined Score:</b> {selected['combined_score']}</p>
        <p><b>Skill Match:</b> {selected['skill_match_pct']}%</p>
        <p><b>Semantic Similarity:</b> {sem_pct_display}%</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- INTERACTIVE CHAT ASSISTANT ----
    st.markdown("## 💬 Interactive Resume Assistant")
    st.session_state.user_q = st.text_input("Ask anything about this candidate:", value=st.session_state.user_q)

    if st.session_state.user_q:
        try:
            with st.spinner("Thinking…"):
                ans = interactive_chatbot(st.session_state.user_q, selected["text"], jd_text)
            st.success(ans)
        except Exception as e:
            st.error(f"OpenAI error: {e}")
            st.write("Fallback advice: Add measurable outcomes, include missing skills, and tailor content to JD keywords.")
else:
    st.info("No screening results yet. Upload resumes and run screening.")
# -------------- END OF FULL APP --------------
