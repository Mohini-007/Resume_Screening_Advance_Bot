# resume_screening.py
import os
from openai import OpenAI
from typing import Optional

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
import re
import tempfile
from typing import List, Optional, Dict, Any

import fitz  # PyMuPDF
import docx2txt
import spacy
from dotenv import load_dotenv

# try imports that may be heavy; handle failures gracefully
try:
    from sentence_transformers import SentenceTransformer, util
    _HAS_SB = True
except Exception:
    _HAS_SB = False

# OpenAI (optional)
try:
    import openai
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

# Load .env (for OPENAI_API_KEY if provided)
load_dotenv()

# spaCy model
nlp = spacy.load("en_core_web_sm")

# ----------------
# Embedding model (lazy load)
# ----------------
_SB_MODEL = None
def load_embedding_model(name: str = "all-MiniLM-L6-v2"):
    global _SB_MODEL
    if not _HAS_SB:
        return None
    if _SB_MODEL is None:
        _SB_MODEL = SentenceTransformer(name)
    return _SB_MODEL

# ----------------
# BASE_SKILLS: 50+ roles
# ----------------
BASE_SKILLS = {
    'Data Analyst': ['Python', 'SQL', 'Excel', 'Tableau', 'Power BI', 'Pandas', 'NumPy', 'Matplotlib', 'Seaborn', 'Statistics', 'Regression', 'Visualization', 'ETL'],
    'Data Engineer': ['Python', 'SQL', 'Spark', 'Airflow', 'Kafka', 'ETL', 'Hadoop', 'Data Modeling', 'Redshift', 'BigQuery'],
    'ML Engineer': ['Python', 'TensorFlow', 'PyTorch', 'Scikit-learn', 'NLP', 'Computer Vision', 'Keras', 'Deep Learning', 'Model Deployment', 'Docker', 'MLflow'],
    'Data Scientist': ['Python', 'R', 'Machine Learning', 'Deep Learning', 'NLP', 'Statistics', 'Pandas', 'NumPy', 'Scikit-learn'],
    'AI Engineer': ['Python', 'NLP', 'Transformers', 'TensorFlow', 'PyTorch', 'OpenCV', 'MLOps'],
    'Cloud Engineer': ['AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'Terraform', 'CI/CD', 'Linux'],
    'DevOps Engineer': ['Linux', 'Shell Scripting', 'Docker', 'Kubernetes', 'Jenkins', 'Terraform', 'CI/CD', 'Monitoring'],
    'Frontend Developer': ['HTML', 'CSS', 'JavaScript', 'React', 'Vue', 'Angular', 'Tailwind', 'Bootstrap', 'TypeScript'],
    'Backend Developer': ['Java', 'Python', 'Node.js', 'SQL', 'NoSQL', 'REST API', 'Spring', 'Django', 'Flask', 'Microservices'],
    'Full Stack Developer': ['JavaScript', 'React', 'Node.js', 'Express', 'MongoDB', 'SQL', 'Docker', 'AWS'],
    'Database Administrator': ['SQL', 'Oracle', 'MySQL', 'PostgreSQL', 'MongoDB', 'Backup', 'Replication', 'Performance Tuning'],
    'Big Data Engineer': ['Hadoop', 'Spark', 'Hive', 'Kafka', 'Scala', 'ETL'],
    'ETL Developer': ['ETL', 'Informatica', 'Talend', 'SSIS', 'SQL', 'Data Warehousing'],
    'Product Manager': ['Roadmaps', 'Agile', 'Scrum', 'User Stories', 'JIRA', 'Business Analysis'],
    'Project Manager': ['Agile', 'Scrum', 'Kanban', 'JIRA', 'Risk Management'],
    'System Administrator': ['Linux', 'Windows Server', 'Networking', 'Shell Scripting', 'Ansible'],
    'AI Researcher': ['Python', 'Deep Learning', 'NLP', 'Reinforcement Learning', 'PyTorch', 'TensorFlow'],
    'Blockchain Developer': ['Solidity', 'Ethereum', 'Smart Contracts', 'Web3.js', 'Hyperledger'],
    'Game Developer': ['Unity', 'C#', 'Unreal Engine', 'C++'],
    'Embedded Systems Engineer': ['C', 'C++', 'RTOS', 'Microcontrollers', 'Verilog'],
    'IoT Engineer': ['Arduino', 'Raspberry Pi', 'Python', 'MQTT'],
    'RPA Developer': ['UiPath', 'Automation Anywhere', 'Blue Prism', 'Python', 'SQL'],
    'Salesforce Developer': ['Apex', 'Visualforce', 'Lightning'],
    'ERP Consultant': ['SAP', 'Oracle ERP'],
    'Financial Analyst': ['Excel', 'SQL', 'Power BI', 'Financial Modeling'],
    'HR Analyst': ['Excel', 'HR Analytics', 'Power BI'],
    'Operations Analyst': ['Excel', 'SQL', 'Process Improvement'],
    'Supply Chain Analyst': ['Excel', 'Tableau', 'Forecasting', 'SQL'],
    'Marketing Analyst': ['Google Analytics', 'Excel', 'Tableau', 'SQL'],
    'SEO Specialist': ['SEO', 'Google Analytics', 'Keyword Research'],
    'Content Writer': ['SEO Writing', 'Copywriting', 'Blog Writing'],
    'Technical Writer': ['Documentation', 'API Writing', 'Technical Content'],
    'Software Tester': ['Selenium', 'JUnit', 'TestNG', 'Manual Testing', 'Automation Testing'],
    'QA Engineer': ['Manual Testing', 'Automation', 'Selenium', 'JMeter'],
    'Mobile App Developer': ['Java', 'Kotlin', 'Swift', 'Flutter', 'React Native'],
    'iOS Developer': ['Swift', 'Xcode'],
    'Android Developer': ['Java', 'Kotlin', 'Android SDK'],
    'UI/UX Designer': ['Figma', 'Adobe XD', 'Sketch'],
    'Network Engineer': ['Networking', 'Cisco', 'Routing', 'Switching', 'Firewall'],
    'Cybersecurity Analyst': ['Network Security', 'Firewalls', 'Penetration Testing', 'SIEM'],
    'Cloud Architect': ['AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'Terraform'],
    'Statistician': ['R', 'Python', 'Statistics', 'Hypothesis Testing'],
    'Research Scientist': ['Python', 'Statistics', 'Machine Learning'],
    'Penetration Tester': ['Kali Linux', 'Burp Suite', 'Metasploit'],
    'IT Consultant': ['Business Analysis', 'Requirement Gathering'],
    'Desktop Support': ['Windows', 'Linux', 'Networking', 'MS Office'],
    'Technical Support Engineer': ['Troubleshooting', 'Customer Support', 'Linux']
}
# ----------------

# ----------------
# Parsing helpers
# ----------------
def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=file_bytes, filetype='pdf')
        text_pages = []
        for page in doc:
            text_pages.append(page.get_text())
        return "\n".join(text_pages)
    except Exception:
        return ""

def extract_text_from_docx_bytes(file_bytes: bytes) -> str:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        text = docx2txt.process(tmp_path)
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return text or ""
    except Exception:
        return ""

def extract_text_from_file(file) -> str:
    # file is a Streamlit UploadedFile-like
    name = getattr(file, "name", "")
    try:
        if name.lower().endswith(".pdf"):
            return extract_text_from_pdf_bytes(file.read())
        elif name.lower().endswith(".docx"):
            return extract_text_from_docx_bytes(file.read())
        else:
            try:
                return file.read().decode("utf-8", errors="ignore")
            except Exception:
                return ""
    except Exception:
        # fallback
        try:
            return file.getvalue().decode("utf-8", errors="ignore")
        except Exception:
            return ""

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

# ----------------
# Skill extraction (simple keyword matching)
# ----------------
def extract_skills(text: str, role_skills: Optional[List[str]] = None) -> List[str]:
    text_low = (text or "").lower()
    found = set()
    candidates = []
    if role_skills:
        candidates = role_skills
    else:
        # flatten baseline
        for v in BASE_SKILLS.values():
            candidates.extend(v)
    # match phrases/words
    for s in candidates:
        if not s:
            continue
        if s.lower() in text_low:
            found.add(s)
    return sorted(found)

# ----------------
# Education extraction (simple heuristics)
# ----------------
def extract_education(text: str) -> List[str]:
    if not text:
        return []
    edu_patterns = [
        r"(b(?:\.|)sc(?:\.|))",
        r"(b(?:\.|)tech)",
        r"bachelor",
        r"master",
        r"m(?:\.|)tech",
        r"m(?:\.|)sc",
        r"mba",
        r"phd"
    ]
    text_low = text.lower()
    found = set()
    for pat in edu_patterns:
        if re.search(pat, text_low):
            found.add(re.search(pat, text_low).group(0))
    # try capturing degree lines (e.g., "Bachelor of Technology")
    lines = text.splitlines()
    for line in lines:
        if any(k in line.lower() for k in ["bachelor", "master", "phd", "mba", "b.tech", "m.tech", "bsc", "msc"]):
            found.add(line.strip())
    return list(found)

# ----------------
# Experience extraction (years + roles)
# ----------------
def extract_years_experience(text: str) -> int:
    if not text:
        return 0
    matches = re.findall(r"(\d+)\+?\s+years?", text.lower())
    nums = []
    for m in matches:
        try:
            nums.append(int(m))
        except Exception:
            pass
    if nums:
        return max(nums)
    # fallback: look for patterns like 'experience: X'
    m2 = re.search(r"experience[:\s]+(\d+)", text.lower())
    if m2:
        try:
            return int(m2.group(1))
        except:
            pass
    return 0

# ----------------
# Semantic similarity & scoring
# ----------------
def calculate_similarity(a: str, b: str) -> float:
    if not _HAS_SB:
        return 0.0
    try:
        model = load_embedding_model()
        emb = model.encode([a, b], convert_to_tensor=True)
        sim = util.cos_sim(emb[0], emb[1]).item()
        return float(max(0.0, min(1.0, sim)))
    except Exception:
        return 0.0

def score_resume(resume_text: str, jd_text: str, role_skills: Optional[List[str]] = None) -> Dict[str, Any]:
    resume_clean = clean_text(resume_text or "")
    jd_clean = clean_text(jd_text or "")
    resume_skills = set(extract_skills(resume_clean, role_skills))
    required = set([s for s in role_skills]) if role_skills else set()
    intersect = required.intersection(resume_skills) if required else set()
    skill_match_pct = (len(intersect) / len(required) * 100) if required else 0.0
    sem = calculate_similarity(resume_clean, jd_clean)
    yrs = extract_years_experience(resume_clean)
    combined = (0.5 * (skill_match_pct/100.0) + 0.4 * sem + 0.1 * min(yrs,10)/10.0) * 100.0
    missing = sorted(list(required - intersect)) if required else []
    return {
        "skill_match_pct": round(skill_match_pct,2),
        "semantic_sim": round(sem,3),
        "years_experience": yrs,
        "combined_score": round(combined,2),
        "resume_skills": sorted(list(resume_skills)),
        "missing_skills": missing
    }

# ----------------
# Rule-based recommendation (fallback)
# ----------------
def generate_recommendation(resume_text: str, jd_text: str, role_skills: Optional[List[str]] = None) -> Dict[str, Any]:
    s = score_resume(resume_text, jd_text, role_skills)
    recs = []
    if s['skill_match_pct'] < 50 and s['missing_skills']:
        recs.append("Add or highlight these missing skills: " + ", ".join(s['missing_skills']))
    else:
        recs.append("Skills match looks decent — prioritize key skills at the top.")
    if s['years_experience'] < 2:
        recs.append("Add internship/project durations & measurable outcomes.")
    recs.append("Use quantifiable achievements and action verbs.")
    recs.append("Tailor bullets to mirror key phrases from the Job Description.")
    summary = f"Overall score: {s['combined_score']} — Skill match: {s['skill_match_pct']}% — Semantic sim: {s['semantic_sim']}"
    return {"summary": summary, "recommendations": recs}

# ----------------
# OpenAI interactive chatbot (if API key set and openai installed)
# ----------------
def interactive_chatbot(user_query: str, resume_text: Optional[str] = None, jd_text: Optional[str] = None) -> str:
    """
    If OPENAI_API_KEY is configured and openai is available, uses GPT to answer.
    Otherwise falls back to a concise rule-based answer constructed from generate_recommendation.
    """
    key = os.getenv("OPENAI_API_KEY")
    if _HAS_OPENAI and key:
        try:
            system_msg = (
                "You are a concise AI career assistant. Provide actionable resume improvement tips. "
                "Be specific and give bullet-style suggestions tailored to the resume & job description."
            )
            user_prompt = user_query
            # include minimal context but avoid sending huge texts—truncate if necessary
            resume_ctx = (resume_text or "")[:4000]
            jd_ctx = (jd_text or "")[:2000]
            full_prompt = f"Job Description:\n{jd_ctx}\n\nResume:\n{resume_ctx}\n\nUser question:\n{user_prompt}"

            # NEW: using OpenAI client correctly
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=350,
                temperature=0.6
            )

            if resp and resp.choices and len(resp.choices) > 0:
                return resp.choices[0].message.content.strip()
            else:
                return "⚠️ OpenAI returned an unexpected response."
        except Exception as e:
            return f"⚠️ OpenAI error: {e}\n\nFallback advice:\n" + "\n".join(
                generate_recommendation(resume_text or "", jd_text or "", role_skills=None)["recommendations"]
            )
    else:
        # fallback to rule-based reply
        rec = generate_recommendation(resume_text or "", jd_text or "", role_skills=None)
        return "Fallback assistant: " + rec["summary"] + " Recommendations: " + " ".join(rec["recommendations"])
