# app.py
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# load env (so OPENAI_API_KEY from .env is available)
load_dotenv()

from resume_screening import (
    extract_text_from_file, clean_text, extract_skills, extract_education,
    extract_years_experience, score_resume, generate_recommendation,
    interactive_chatbot, BASE_SKILLS, calculate_similarity
)

st.set_page_config(page_title="Resume Screening Bot", layout="wide")
st.title("📄 Resume Screening Bot")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    roles = ["All"] + sorted(list(BASE_SKILLS.keys()))
    selected_role = st.selectbox("Filter by role", roles)
    top_k = st.number_input("Top K resumes to save", min_value=1, max_value=50, value=5)
    save_top = st.checkbox("Save top resumes to folder", value=True)
    st.markdown("---")
    st.caption("Tools: Python, spaCy, SentenceTransformers, PyMuPDF, Pandas, Streamlit")

# Upload / JD
uploaded_files = st.file_uploader("Upload resumes (PDF/DOCX) — multiple allowed", accept_multiple_files=True, type=['pdf','docx'])
jd_text = st.text_area("Paste Job Description here", height=220)
process_btn = st.button("Process Resumes")

# storage for results in session
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "raw_results" not in st.session_state:
    st.session_state.raw_results = []

if process_btn:
    if not uploaded_files:
        st.warning("Upload at least one resume file.")
    elif not jd_text.strip():
        st.warning("Paste a Job Description to match against.")
    else:
        os.makedirs("top_resumes", exist_ok=True)
        data = []
        role_skills = BASE_SKILLS.get(selected_role) if selected_role != "All" else None

        progress = st.progress(0)
        total = len(uploaded_files)
        for i, f in enumerate(uploaded_files):
            fname = f.name
            # read and parse
            text = extract_text_from_file(f)
            text = clean_text(text)
            edu = extract_education(text)
            yrs = extract_years_experience(text)
            skills = extract_skills(text, role_skills)
            sc = score_resume(text, jd_text, role_skills)
            rec = generate_recommendation(text, jd_text, role_skills)

            row = {
                "filename": fname,
                "text_trunc": (text or "")[:5000],
                "education": ", ".join(edu) if edu else "Not found",
                "years_experience": yrs,
                "skills": ", ".join(skills) if skills else "",
                "skill_match_pct": sc["skill_match_pct"],
                "semantic_sim": sc["semantic_sim"],
                "combined_score": sc["combined_score"],
                "missing_skills": ", ".join(sc.get("missing_skills", [])),
                "recommendation_summary": rec["summary"],
                "recommendations": " | ".join(rec["recommendations"]),
                "file_obj": f  # keep file object for saving top resumes
            }
            data.append(row)
            progress.progress((i+1)/total)

        df = pd.DataFrame(data).sort_values(by="combined_score", ascending=False).reset_index(drop=True)
        st.session_state.results_df = df
        st.session_state.raw_results = data

        # Save top k
        if save_top:
            for idx, r in df.head(top_k).iterrows():
                try:
                    with open(os.path.join("top_resumes", r["filename"]), "wb") as out:
                        out.write(r["file_obj"].getbuffer())
                except Exception:
                    pass
            st.success(f"Top {top_k} resumes saved to 'top_resumes/'")

# Show results if present
df = st.session_state.get("results_df")
if df is not None:
    st.subheader("📊 Candidate Rankings")
    display_df = df[["filename", "combined_score", "skill_match_pct", "semantic_sim", "years_experience", "education", "skills"]]
    st.dataframe(display_df)

    # Chart
    st.subheader("Top Candidates — Score Chart")
    top_show = df.head(10)
    fig, ax = plt.subplots(figsize=(10,4))
    ax.bar(top_show['filename'], top_show['combined_score'])
    ax.set_xlabel("Candidate")
    ax.set_ylabel("Combined Score")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    # CSV download
    csv = df.drop(columns=["file_obj", "text_trunc"]).to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV Report", csv, "candidate_rankings.csv", mime="text/csv")

    # Candidate inspector
    st.subheader("🎯 Inspect Candidate")
    candidate = st.selectbox("Choose candidate", df["filename"].tolist())
    cand = df[df["filename"] == candidate].iloc[0]
    st.markdown(f"**File:** {cand['filename']}")
    st.markdown(f"**Combined Score:** {cand['combined_score']} — **Skill match:** {cand['skill_match_pct']}% — **Semantic sim:** {cand['semantic_sim']}")
    st.markdown(f"**Years experience:** {cand['years_experience']}")
    st.markdown(f"**Education:** {cand['education']}")
    st.markdown("**Extracted skills:**")
    st.write(cand["skills"] or "None detected")
    st.markdown("**Rule-based recommendations:**")
    st.write(cand["recommendations"])

    # Show parsed text (optional)
    if st.checkbox("Show parsed resume text (truncated)"):
        st.text_area("Parsed text", value=cand["text_trunc"], height=300)

    # Interactive chatbot (candidate-aware)
    st.subheader("💬 Interactive Resume Assistant (AI-powered if configured)")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_q = st.text_input("Ask the assistant about this candidate (e.g., 'What highlights to add?')")
    if st.button("Ask AI") and user_q:
        # candidate-aware context
        try:
            # rebuild full resume text from stored object if available
            full_resume_text = ""
            try:
                fobj = cand["file_obj"]
                full_resume_text = extract_text_from_file(fobj)
            except Exception:
                full_resume_text = cand.get("text_trunc","")
            answer = interactive_chatbot(user_q, resume_text=full_resume_text, jd_text=jd_text)
        except Exception as e:
            answer = f"⚠️ Chatbot error: {e}\n\nFallback advice:\n" + " ".join(generate_recommendation(full_resume_text, jd_text)["recommendations"])

        st.session_state.chat_history.append(("You", user_q))
        st.session_state.chat_history.append(("Assistant", answer))

    # show chat history
    for sender, msg in st.session_state.chat_history:
        if sender == "You":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Assistant:** {msg}")

else:
    st.info("No results yet — upload resumes and paste a Job Description, then click 'Process Resumes'.")
