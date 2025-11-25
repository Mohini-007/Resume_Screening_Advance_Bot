# Growfinix-Tech
Data Science Internship tasks &amp; projects at Growfinix Technology.

# 📄 Resume Screening Bot using NLP & AI

## Project Overview

This project automates resume screening and candidate ranking using **Natural Language Processing (NLP)** and **AI-powered insights**. It helps recruiters quickly evaluate resumes against job descriptions, extract skills, experience, and education, and provide actionable recommendations.

---

## Features

* ✅ Upload and parse multiple resumes (PDF/DOCX)
* ✅ Extract structured text: skills, education, experience
* ✅ Match resumes to Job Descriptions using **semantic similarity**
* ✅ Rank candidates by **skill match**, **semantic similarity**, and **experience**
* ✅ Role-based filtering (e.g., Data Analyst, ML Engineer)
* ✅ Display match score and skill comparison charts
* ✅ Download ranked candidate CSV reports
* ✅ Save top resumes to a separate folder for easy access
* ✅ **Bonus:** AI-powered interactive chatbot for personalized resume improvement suggestions

---

## Technologies Used

* **Python:** Core language
* **spaCy:** NLP processing
* **Sentence Transformers:** Semantic similarity
* **PyMuPDF / docx2txt:** Resume parsing
* **Pandas / Matplotlib:** Data processing and visualization
* **Streamlit:** Interactive web app UI
* **OpenAI GPT API:** AI-driven resume recommendations

---

## How to Run

```bash
# 1. Clone the repository
git clone <your-repo-link>
cd <repo-folder>

# 2. Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set OpenAI API key (optional for AI recommendations)
export OPENAI_API_KEY="your_api_key"  # Linux/macOS
setx OPENAI_API_KEY "your_api_key"    # Windows

# 5. Run the Streamlit app
streamlit run bot.py
```

---

## Interactive Features

* **Upload Resumes:** Drag and drop multiple files (PDF/DOCX)
* **Paste Job Description:** Text area for JD input
* **Candidate Ranking:** Table and bar chart visualization
* **Top Resumes:** Automatically saved to `top_resumes/` folder
* **AI Assistant:** Ask targeted questions about resume improvements

---

## Repository Structure

```
Resume-Screening-Bot/
│
├─ bot.py                  # Streamlit frontend
├─ resume_screening.py     # Resume parsing, scoring, recommendations
├─ requirements.txt        # Python dependencies
├─ top_resumes/            # Folder for top resumes
├─ README.md               # Project documentation
```

---

## Author

**Mohini Bajarang Fulsagar**
* B.Tech (CSE), 2025
* AI, NLP, Data Science Projects
* LinkedIn: https://www.linkedin.com/in/mohini-fulsagar-263100246
