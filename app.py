import streamlit as st
import pandas as pd
import numpy as np
import PyPDF2
import re
from io import BytesIO
from reportlab.pdfgen import canvas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import spacy

# ------------------ NLTK & Spacy Setup ------------------
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")

# ------------------ Sidebar ------------------
st.sidebar.title("📝 Resume Analyzer Pro")
theme = st.sidebar.radio("Theme", ["Light","Dark"])
if theme == "Dark":
    st.markdown("""
    <style>
    .reportview-container {background-color: #0E1117; color: white;}
    </style>
    """, unsafe_allow_html=True)

# ------------------ Resume Upload ------------------
uploaded_files = st.file_uploader(
    "Upload Resume(s) PDF/TXT (DOCX not supported)", accept_multiple_files=True
)

job_desc = st.text_area("Paste Job Description Here")

# ------------------ Functions ------------------
def extract_text(file):
    text = ""
    if file.name.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    elif file.name.endswith(".txt"):
        text = str(file.read(), "utf-8")
    else:
        st.warning(f"{file.name} format not supported. Only PDF/TXT allowed.")
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_skills(text):
    skills_db = ["python","java","c++","sql","machine learning","deep learning","nlp","excel","html","css","javascript","aws","docker","git"]
    found_skills = [skill for skill in skills_db if skill.lower() in text.lower()]
    return found_skills

def keyword_match(resume_text, job_desc):
    resume_tokens = [w.lower() for w in nltk.word_tokenize(resume_text) if w.isalnum()]
    jd_tokens = [w.lower() for w in nltk.word_tokenize(job_desc) if w.isalnum()]
    match_count = len(set(resume_tokens) & set(jd_tokens))
    return round((match_count / len(set(jd_tokens))) * 100 if jd_tokens else 0, 2)

def generate_pdf_report(candidate):
    buffer = BytesIO()
    c = canvas.Canvas(buffer)
    c.drawString(100, 800, f"Candidate: {candidate['Name']}")
    c.drawString(100, 780, f"Skills Found: {', '.join(candidate['Skills'])}")
    c.drawString(100, 760, f"Keyword Match: {candidate['Keyword Match %']}%")
    missing_skills = candidate.get('Missing Skills', [])
    c.drawString(100, 740, f"Missing Skills: {', '.join(missing_skills) if missing_skills else 'None'}")
    c.save()
    return buffer

# ------------------ Process Resumes ------------------
candidate_data = []

if uploaded_files:
    for file in uploaded_files:
        text = extract_text(file)
        if text.strip() == "":
            continue
        skills = extract_skills(text)
        match_score = keyword_match(text, job_desc)
        candidate_data.append({
            "Name": file.name,
            "Skills": skills,
            "Skill Count": len(skills),
            "Keyword Match %": match_score,
            "Resume Text": text
        })

    if candidate_data:
        df = pd.DataFrame(candidate_data)
        st.subheader("Candidate Overview")
        st.dataframe(df[["Name","Skills","Skill Count","Keyword Match %"]])

        # Skill Gap Analysis
        st.subheader("Skill Gap Analysis")
        all_skills_needed = [w.lower() for w in nltk.word_tokenize(job_desc) if w.isalnum()]
        for idx, row in df.iterrows():
            missing_skills = list(set(all_skills_needed) - set([s.lower() for s in row["Skills"]]))
            df.at[idx, 'Missing Skills'] = missing_skills
            st.write(f"**{row['Name']}** missing skills: {missing_skills if missing_skills else 'None'}")

        # AI Explainability (TF-IDF + Cosine similarity)
        st.subheader("AI Explainability (TF-IDF + Cosine Similarity)")
        tfidf = TfidfVectorizer()
        vectors = tfidf.fit_transform([job_desc] + df["Resume Text"].tolist())
        cos_sim = cosine_similarity(vectors[0:1], vectors[1:])
        for i, sim in enumerate(cos_sim[0]):
            st.write(f"{df['Name'][i]} similarity: {round(sim*100,2)}%")

        # Download PDF Reports
        st.subheader("Download Candidate Reports")
        for idx, row in df.iterrows():
            buffer = generate_pdf_report(row)
            st.download_button(
                label=f"Download {row['Name']} Report",
                data=buffer,
                file_name=f"{row['Name']}_report.pdf",
                mime="application/pdf"
            )

# ------------------ Placeholder Sections ------------------
st.subheader("Fairness & Bias Dashboard (Example)")
st.write("Gender/Age/Education bias metrics can be calculated here using AI Fairness 360")

st.subheader("Career Path & Skill Clustering (Example)")
st.write("Suggest next skills or career growth paths based on current skills")

st.subheader("Interactive What-If Analysis")
st.write("Modify candidate skills or experience and see AI scoring changes (future implementation)")
