"""
matcher.py — Core AI logic for ResumeIQ
========================================
This file handles:
 1. Extracting text from a resume PDF
 2. Cleaning and preprocessing text
 3. Computing whole-document semantic similarity (Sentence Transformers)
 4. Detecting skills using BOTH exact keyword matching AND semantic
    (embedding) matching, so paraphrased skills ("built ML models") are
    still detected even when the exact skill name isn't present
 5. Producing a composite, explainable match score with evidence
    (which sentence triggered each skill match, and how confident we are)

Design notes / why this changed from v1:
 - v1 only did exact keyword matching for skills, which meant "developed
   machine learning systems" would NOT count as "machine learning" unless
   that literal phrase appeared. This version adds a semantic fallback so
   skills are detected by meaning, not just wording.
 - v1's score was ONLY the whole-document cosine similarity. That's easy to
   game (a long, well-written but irrelevant resume can score deceptively
   high). This version blends document similarity with actual skill
   coverage into one composite score.
 - Heavy ML imports (sentence-transformers) are done LAZILY inside the
   functions that need them, not at module import time. This keeps the
   module importable and unit-testable without installing torch, and lets
   exact-match-only skill detection run with model=None.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import pdfplumber

logger = logging.getLogger("resumeiq.matcher")

# ── Skill library ──────────────────────────────────────────────────────────
SKILL_KEYWORDS = [
    # Programming languages
    "python", "java", "c++", "javascript", "typescript", "go", "rust", "kotlin",
    "swift", "r", "scala", "sql", "bash", "shell",
    # ML / AI
    "machine learning", "deep learning", "neural network", "nlp",
    "computer vision", "reinforcement learning", "llm", "transformer",
    "pytorch", "tensorflow", "keras", "scikit-learn", "huggingface",
    "langchain", "openai", "bert", "gpt",
    # Data & Cloud
    "pandas", "numpy", "matplotlib", "seaborn", "spark", "hadoop",
    "aws", "gcp", "azure", "docker", "kubernetes", "mlflow",
    # Web / Software
    "react", "node.js", "django", "flask", "fastapi", "rest api",
    "graphql", "git", "ci/cd", "agile", "microservices",
    # Soft skills / concepts
    "communication", "leadership", "problem solving", "teamwork",
    "data structures", "algorithms", "system design", "object oriented",
]

# Weight given to whole-document semantic similarity vs. actual skill
# coverage when computing the final composite score.
DOC_SIMILARITY_WEIGHT = 0.6
SKILL_COVERAGE_WEIGHT = 0.4

# Minimum cosine similarity for a sentence to count as a semantic
# (paraphrased) match for a skill that wasn't found via exact keyword match.
SEMANTIC_SKILL_THRESHOLD = 0.55

MAX_PDF_CHARS = 50_000  # sanity cap so a malformed/huge PDF can't blow up the UI


class MatcherError(Exception):
    """Raised for user-facing, expected failure modes (bad PDF, empty input)."""


def extract_text_from_pdf(pdf_file) -> str:
    """
    Reads a PDF file (file-like object or path) and returns all text.
    Uses pdfplumber, which handles multi-column layouts better than basic
    PDF readers.
    """
    text_parts = []
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    except Exception as exc:  # pdfplumber raises various low-level exceptions
        raise MatcherError(f"Could not read PDF: {exc}") from exc

    text = "\n".join(text_parts).strip()
    if len(text) > MAX_PDF_CHARS:
        logger.warning("Resume text truncated from %d to %d chars", len(text), MAX_PDF_CHARS)
        text = text[:MAX_PDF_CHARS]
    return text


def clean_text(text: str) -> str:
    """Lowercases, strips special characters, collapses whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s.,\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    """Best-effort sentence/line splitter used for semantic evidence lookup."""
    raw = re.split(r"[\n.]+", text)
    sentences = [s.strip() for s in raw if len(s.strip()) > 3]
    return sentences or ([text.strip()] if text.strip() else [])


def extract_skills(text: str, model=None, threshold: float = SEMANTIC_SKILL_THRESHOLD) -> dict:
    """
    Detects skills in `text` two ways:
      1. Exact keyword match (fast, always runs) — full confidence, evidence
         is the sentence containing the literal phrase.
      2. Semantic match (only for skills not already found exactly, only if
         a model is provided) — embeds each remaining skill phrase and each
         sentence in the text, and counts a match if the best cosine
         similarity clears `threshold`.

    Returns: {skill_name: {"matched_via": "exact"|"semantic",
                            "evidence": str, "confidence": float}}

    Passing model=None disables the semantic pass entirely (used in tests
    and any environment without the embedding model loaded) — you still get
    full exact-match detection.
    """
    text_lower = text.lower()
    sentences = _split_sentences(text)
    results: dict[str, dict] = {}
    remaining = []

    for skill in SKILL_KEYWORDS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        match = re.search(pattern, text_lower)
        if match:
            evidence = next((s for s in sentences if skill in s.lower()), skill)
            results[skill] = {"matched_via": "exact", "evidence": evidence, "confidence": 1.0}
        else:
            remaining.append(skill)

    if model is not None and remaining and sentences:
        from sentence_transformers import util  # lazy import — see module docstring

        skill_embeddings = model.encode(remaining, convert_to_tensor=True)
        sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
        sims = util.cos_sim(skill_embeddings, sentence_embeddings)

        for i, skill in enumerate(remaining):
            best_idx = int(sims[i].argmax())
            best_score = float(sims[i][best_idx])
            if best_score >= threshold:
                results[skill] = {
                    "matched_via": "semantic",
                    "evidence": sentences[best_idx],
                    "confidence": round(best_score, 2),
                }

    return results


def compute_match_score(resume_text: str, job_text: str, model) -> float:
    """
    Whole-document semantic similarity between resume and job description,
    using Sentence Transformer embeddings + cosine similarity. Returns a
    float in [0, 1]. This is ONE input into the final composite score
    returned by generate_report — see that function for why it's not used
    alone.
    """
    from sentence_transformers import util  # lazy import — see module docstring

    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    job_embedding = model.encode(job_text, convert_to_tensor=True)
    similarity = util.cos_sim(resume_embedding, job_embedding)
    score = float(similarity[0][0])
    return max(0.0, min(1.0, score))


def generate_report(resume_text: str, job_text: str, doc_score: float, model=None) -> dict:
    """
    Produces the full explainable report:
      - A composite score blending whole-document similarity and actual
        skill coverage (not just document similarity alone — see module
        docstring for why that was a flaw in v1)
      - Skills you HAVE that the job wants, each with evidence of *why*
        we think you have it (exact phrase or paraphrase + confidence)
      - Skills the job wants but you're missing
      - Skills on your resume not mentioned in the job (bonus skills)
      - A plain-English recommendation
    """
    resume_skills = extract_skills(resume_text, model=model)
    job_skills = extract_skills(job_text, model=model)

    resume_set = set(resume_skills)
    job_set = set(job_skills)

    matched_names = sorted(resume_set & job_set)
    missing_names = sorted(job_set - resume_set)
    extra_names = sorted(resume_set - job_set)

    matched_skills = [
        {"skill": s, **resume_skills[s]} for s in matched_names
    ]
    missing_skills = [
        {"skill": s, **job_skills[s]} for s in missing_names
    ]
    extra_skills = [
        {"skill": s, **resume_skills[s]} for s in extra_names
    ]

    if job_set:
        skill_coverage = len(matched_names) / len(job_set)
        composite = (DOC_SIMILARITY_WEIGHT * doc_score) + (SKILL_COVERAGE_WEIGHT * skill_coverage)
    else:
        # No recognizable skills in the job description at all — there's
        # nothing to measure coverage against, so don't let a vacuous "100%
        # coverage" default inflate the score. Fall back to document
        # similarity alone.
        skill_coverage = 1.0
        composite = doc_score
    composite = max(0.0, min(1.0, composite))
    pct = round(composite * 100, 1)

    if pct >= 80:
        recommendation = (
            "🟢 Excellent match! Your resume strongly aligns with this role. "
            "Focus on tailoring your cover letter and highlighting your key projects."
        )
    elif pct >= 60:
        recommendation = (
            "🟡 Good match with room to improve. Consider adding the missing skills "
            "to your resume if you have experience with them, or mention them in your cover letter."
        )
    elif pct >= 40:
        recommendation = (
            "🟠 Moderate match. The role requires several skills not visible on your resume. "
            "Consider upskilling in the missing areas before applying."
        )
    else:
        recommendation = (
            "🔴 Low match. This role may not align well with your current profile. "
            "Focus on roles that better match your existing skills, or invest time "
            "in building the missing competencies first."
        )

    return {
        "score_pct": pct,
        "doc_similarity_pct": round(doc_score * 100, 1),
        "skill_coverage_pct": round(skill_coverage * 100, 1),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "extra_skills": extra_skills,
        "recommendation": recommendation,
        "resume_skill_count": len(resume_set),
        "job_skill_count": len(job_set),
    }
