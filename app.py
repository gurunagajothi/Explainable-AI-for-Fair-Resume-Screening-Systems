import streamlit as st
import matplotlib.pyplot as plt
import PyPDF2
import re
import nltk
import numpy as np

from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ==============================
# NLTK
# ==============================
nltk.download("stopwords")

# ==============================
# Load SBERT model (semantic AI)
# ==============================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ==============================
# Skill Weights (ATS-grade)
# ==============================
SKILL_WEIGHTS = {
    "python": 5,
    "java": 4,
    "javascript": 5,
    "react": 5,
    "node": 5,
    "express": 4,
    "mysql": 4,
    "docker": 3,
    "aws": 3,
    "rest": 4,
    "api": 4,
    "dsa": 5,
    "data": 4,
    "structures": 4,
    "algorithms": 4,
    "machine": 3,
    "learning": 3,
    "ai": 3,
    "automation": 3
}

# ==============================
# Page Config
# ==============================
st.set_page_config(
    page_title="AI Resume Analyzer (Level 2 ATS)",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume Analyzer – Level 2 (IIT/NIT Grade)")
st.write("Semantic + Skill-weighted ATS Resume Matching")

# ==============================
# Helper Functions
# ==============================
def extract_text_from_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        return " ".join(page.extract_text() for page in reader.pages)
    except:
        return ""

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def remove_stopwords(text):
    sw = set(stopwords.words("english"))
    return " ".join(w for w in text.split() if w not in sw)

def semantic_similarity(text1, text2):
    emb1 = model.encode(text1)
    emb2 = model.encode(text2)
    return cosine_similarity([emb1], [emb2])[0][0] * 100

def extract_weighted_skills(text):
    words = set(text.split())
    return {skill: weight for skill, weight in SKILL_WEIGHTS.items() if skill in words}

def skill_match_score(resume_text, job_text):
    resume_skills = extract_weighted_skills(resume_text)
    job_skills = extract_weighted_skills(job_text)

    if not job_skills:
        return 0

    matched_weight = sum(
        weight for skill, weight in job_skills.items()
        if skill in resume_skills
    )
    total_weight = sum(job_skills.values())

    return (matched_weight / total_weight) * 100

# ==============================
# ATS Score Calculation
# ==============================
def calculate_ats_score(resume_text, job_text):
    resume_clean = remove_stopwords(clean_text(resume_text))
    job_clean = remove_stopwords(clean_text(job_text))

    semantic_score = semantic_similarity(resume_clean, job_clean)
    skill_score = skill_match_score(resume_clean, job_clean)

    # ATS Final Score (industry-like weighting)
    final_score = (semantic_score * 0.4) + (skill_score * 0.6)

    return round(final_score, 2), round(semantic_score, 2), round(skill_score, 2)

# ==============================
# UI
# ==============================
resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_desc = st.text_area("Paste Job Description", height=200)

if st.button("Analyze Resume"):
    if not resume_file or not job_desc.strip():
        st.warning("Please upload resume and paste job description")
    else:
        resume_text = extract_text_from_pdf(resume_file)

        final_score, semantic_score, skill_score = calculate_ats_score(
            resume_text, job_desc
        )

        st.subheader("📊 ATS Match Analytics")

        c1, c2, c3 = st.columns(3)
        c1.metric("Final ATS Score", f"{final_score}%")
        c2.metric("Semantic Similarity", f"{semantic_score}%")
        c3.metric("Skill Match", f"{skill_score}%")

        fig, ax = plt.subplots(figsize=(6, 0.5))
        ax.barh([0], [final_score])
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xlabel("ATS Match %")
        ax.set_title("Resume vs Job Description")
        st.pyplot(fig)
