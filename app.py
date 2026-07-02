"""
app.py — Streamlit Web UI for ResumeIQ
========================================
Run with:  streamlit run app.py

What this file does:
 - Creates the entire web interface users see in their browser
 - Accepts resume PDF upload + job description text
 - Calls matcher.py for all AI logic
 - Displays an explainable, visual report with evidence for every skill match
"""

import logging

import streamlit as st
from sentence_transformers import SentenceTransformer

from matcher import (
    MatcherError,
    clean_text,
    compute_match_score,
    extract_text_from_pdf,
    generate_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("resumeiq.app")

MAX_UPLOAD_MB = 5
MIN_JOB_DESC_CHARS = 30

# ── Page configuration (must be the very first Streamlit call) ────────────────
st.set_page_config(
    page_title="ResumeIQ — AI Job Match Scorer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS styling ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #e8eaf0; }
    .hero-title {
        font-size: 3rem; font-weight: 800;
        background: linear-gradient(135deg, #4f8ef7, #a78bfa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.2rem;
    }
    .hero-sub { text-align: center; color: #9ca3af; font-size: 1.1rem; margin-bottom: 2.5rem; }
    .score-box { background: #1c1f2e; border-radius: 16px; padding: 2rem; text-align: center; border: 1px solid #2d3148; }
    .score-number { font-size: 4rem; font-weight: 900; line-height: 1; }
    .score-label { color: #9ca3af; font-size: 0.9rem; margin-top: 0.3rem; }
    .pill-green { display: inline-block; background: #14532d; color: #86efac; padding: 4px 12px; border-radius: 20px; font-size: 0.82rem; margin: 3px; cursor: help; }
    .pill-red { display: inline-block; background: #450a0a; color: #fca5a5; padding: 4px 12px; border-radius: 20px; font-size: 0.82rem; margin: 3px; cursor: help; }
    .pill-blue { display: inline-block; background: #1e3a5f; color: #93c5fd; padding: 4px 12px; border-radius: 20px; font-size: 0.82rem; margin: 3px; cursor: help; }
    .pill-tag { font-size: 0.65rem; opacity: 0.75; margin-left: 4px; }
    .section-card { background: #1c1f2e; border-radius: 12px; padding: 1.5rem; border: 1px solid #2d3148; margin-bottom: 1rem; }
    .section-title { font-size: 1rem; font-weight: 700; margin-bottom: 0.8rem; color: #e8eaf0; }
    .breakdown-row { display: flex; justify-content: space-between; padding: 4px 0; font-size: 0.85rem; color: #9ca3af; }
    .stFileUploader > div { background: #1c1f2e; border-radius: 12px; }
    .stTextArea textarea { background: #1c1f2e !important; color: #e8eaf0 !important; border: 1px solid #2d3148 !important; border-radius: 8px !important; }
    .stButton > button { background: linear-gradient(135deg, #4f8ef7, #a78bfa); color: white; border: none; border-radius: 10px; padding: 0.7rem 2rem; font-size: 1rem; font-weight: 700; width: 100%; cursor: pointer; transition: opacity 0.2s; }
    .stButton > button:hover { opacity: 0.85; }
    .stProgress > div > div { background: linear-gradient(135deg, #4f8ef7, #a78bfa); }
    #MainMenu { visibility: hidden; } footer { visibility: hidden; } header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model():
    """
    all-MiniLM-L6-v2 is a small, fast, high-quality sentence embedding model.
    First run downloads ~90MB; cached locally after that.
    """
    return SentenceTransformer("all-MiniLM-L6-v2")


def render_pills(items, css_class):
    """items: list of {'skill', 'matched_via', 'evidence', 'confidence'}"""
    spans = []
    for item in items:
        tag = "≈" if item.get("matched_via") == "semantic" else ""
        evidence = (item.get("evidence") or "").replace('"', "'")[:160]
        conf = item.get("confidence", 1.0)
        title = f"{evidence}  (confidence: {conf})"
        spans.append(
            f'<span class="{css_class}" title="{title}">{item["skill"]}'
            f'<span class="pill-tag">{tag}</span></span>'
        )
    return "".join(spans)


# ── Hero Header ────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🎯 ResumeIQ</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">AI-powered Resume × Job Description match scorer with explainability</div>',
    unsafe_allow_html=True,
)

# ── Input Section ──────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.markdown("### 📄 Upload Your Resume")
    st.caption(f"PDF format only, up to {MAX_UPLOAD_MB}MB. Your file is processed locally — never stored.")
    resume_file = st.file_uploader(label="resume_upload", type=["pdf"], label_visibility="collapsed")

with col_right:
    st.markdown("### 💼 Paste Job Description")
    st.caption("Copy the full job description from LinkedIn, Google Careers, etc.")
    job_desc = st.text_area(
        label="job_desc_input",
        placeholder="Paste the job description here...\n\nExample:\nWe are looking for a Software Engineer with experience in Python, machine learning, and distributed systems...",
        height=200,
        label_visibility="collapsed",
    )

st.markdown("<br>", unsafe_allow_html=True)
analyze_btn = st.button("🚀 Analyze My Match", use_container_width=True)


# ── Analysis + Results ─────────────────────────────────────────────────────────
if analyze_btn:
    if not resume_file:
        st.error("⚠️ Please upload your resume PDF.")
        st.stop()

    if resume_file.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"⚠️ File too large. Please upload a PDF under {MAX_UPLOAD_MB}MB.")
        st.stop()

    if len(job_desc.strip()) < MIN_JOB_DESC_CHARS:
        st.error(f"⚠️ Please paste a fuller job description (at least {MIN_JOB_DESC_CHARS} characters).")
        st.stop()

    with st.spinner("🤖 AI is analyzing your resume..."):
        model = load_model()

        try:
            resume_raw = extract_text_from_pdf(resume_file)
        except MatcherError as e:
            st.error(f"⚠️ {e}. Try a different PDF.")
            logger.warning("PDF extraction failed: %s", e)
            st.stop()

        if len(resume_raw) < 50:
            st.error("⚠️ PDF appears to be empty or image-based (scanned). Please use a text-based PDF.")
            st.stop()

        resume_clean = clean_text(resume_raw)
        job_clean = clean_text(job_desc)

        doc_score = compute_match_score(resume_clean, job_clean, model)
        report = generate_report(resume_clean, job_clean, doc_score, model=model)

        logger.info(
            "Analysis complete: score=%s%% resume_chars=%d job_chars=%d",
            report["score_pct"], len(resume_raw), len(job_desc),
        )

    # ── Display Results ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📊 Your Match Report")

    pct = report["score_pct"]
    color = "#22c55e" if pct >= 80 else "#eab308" if pct >= 60 else "#f97316" if pct >= 40 else "#ef4444"

    top_left, top_mid, top_right = st.columns([1, 1, 1])

    with top_left:
        st.markdown(f"""
        <div class="score-box">
            <div class="score-number" style="color:{color}">{pct}%</div>
            <div class="score-label">Overall Match Score</div>
        </div>""", unsafe_allow_html=True)

    with top_mid:
        st.markdown(f"""
        <div class="score-box">
            <div class="score-number" style="color:#4f8ef7">{len(report['matched_skills'])}</div>
            <div class="score-label">Skills Matched</div>
        </div>""", unsafe_allow_html=True)

    with top_right:
        st.markdown(f"""
        <div class="score-box">
            <div class="score-number" style="color:#f97316">{len(report['missing_skills'])}</div>
            <div class="score-label">Skills to Add</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.progress(pct / 100)

    with st.expander("ℹ️ How this score is calculated"):
        st.markdown(f"""
        The overall score blends two signals so a long-but-irrelevant resume can't
        game the ranking:
        - **Document semantic similarity** (whole resume vs. whole job description): **{report['doc_similarity_pct']}%**
        - **Skill coverage** (share of the job's required skills you actually have): **{report['skill_coverage_pct']}%**
        - **Composite** = 60% similarity + 40% skill coverage = **{pct}%**

        Skill pills marked **≈** were detected semantically (a paraphrase, not the
        exact wording) — hover over any pill to see the evidence sentence and
        confidence score.
        """)

    st.info(report["recommendation"])

    st.markdown("<br>", unsafe_allow_html=True)
    skill_col1, skill_col2, skill_col3 = st.columns(3)

    with skill_col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">✅ Your Matching Skills</div>', unsafe_allow_html=True)
        if report["matched_skills"]:
            st.markdown(render_pills(report["matched_skills"], "pill-green"), unsafe_allow_html=True)
        else:
            st.caption("No matching skills detected. Try expanding your resume keywords.")
        st.markdown('</div>', unsafe_allow_html=True)

    with skill_col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">❌ Missing Skills (Job Wants These)</div>', unsafe_allow_html=True)
        if report["missing_skills"]:
            st.markdown(render_pills(report["missing_skills"], "pill-red"), unsafe_allow_html=True)
        else:
            st.caption("🎉 No missing skills detected!")
        st.markdown('</div>', unsafe_allow_html=True)

    with skill_col3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">💡 Your Bonus Skills</div>', unsafe_allow_html=True)
        st.caption("Skills you have that aren't explicitly required:")
        if report["extra_skills"]:
            st.markdown(render_pills(report["extra_skills"], "pill-blue"), unsafe_allow_html=True)
        else:
            st.caption("None detected beyond job requirements.")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("🔍 View Extracted Resume Text"):
        st.text(resume_raw[:3000] + ("..." if len(resume_raw) > 3000 else ""))


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center;color:#4b5563;font-size:0.8rem;">'
    'ResumeIQ • Built with Sentence Transformers + Streamlit • '
    'All processing is local — your data never leaves your machine'
    '</p>',
    unsafe_allow_html=True,
)
