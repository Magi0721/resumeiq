"""
Unit tests for matcher.py.

These tests deliberately run WITHOUT the sentence-transformers model
(model=None everywhere) so they're fast and don't require downloading any
ML weights or having internet access — they verify all the logic that
doesn't depend on embeddings: text cleaning, exact skill detection, report
composition, and score-bucket recommendations.

Run with:  pytest tests/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from matcher import clean_text, extract_skills, generate_report  # noqa: E402


def test_clean_text_lowercases_and_strips_special_chars():
    raw = "Python  Developer!!  \n\n With ML/AI Experience???"
    cleaned = clean_text(raw)
    assert cleaned == cleaned.lower()
    assert "!" not in cleaned
    assert "  " not in cleaned  # whitespace collapsed


def test_clean_text_handles_empty_string():
    assert clean_text("") == ""


def test_extract_skills_exact_match():
    text = "I have 3 years of experience with python and react."
    skills = extract_skills(text, model=None)
    assert "python" in skills
    assert "react" in skills
    assert skills["python"]["matched_via"] == "exact"
    assert skills["python"]["confidence"] == 1.0


def test_extract_skills_word_boundaries_avoid_false_positives():
    # "r" should not match inside "transformer" or "developer"
    text = "I built a transformer-based developer tool."
    skills = extract_skills(text, model=None)
    assert "r" not in skills


def test_extract_skills_no_semantic_pass_without_model():
    # A paraphrase with no exact keyword should NOT be found when model=None
    text = "I developed systems that learn from data automatically."
    skills = extract_skills(text, model=None)
    assert "machine learning" not in skills


def test_generate_report_structure_and_recommendation_bucket():
    resume = "python developer with react and docker experience"
    job = "looking for a python engineer who knows react, docker, and aws"

    report = generate_report(resume, job, doc_score=0.9, model=None)

    assert 0 <= report["score_pct"] <= 100
    assert "matched_skills" in report
    assert "missing_skills" in report
    assert "extra_skills" in report
    assert isinstance(report["matched_skills"], list)

    matched_names = {s["skill"] for s in report["matched_skills"]}
    missing_names = {s["skill"] for s in report["missing_skills"]}
    assert "python" in matched_names
    assert "react" in matched_names
    assert "aws" in missing_names

    assert report["score_pct"] >= 80  # high doc_score + strong skill overlap
    assert "Excellent" in report["recommendation"] or "🟢" in report["recommendation"]


def test_generate_report_low_score_bucket():
    report = generate_report("no relevant experience here", "totally unrelated field text", doc_score=0.05, model=None)
    assert report["score_pct"] < 40
    assert "🔴" in report["recommendation"]


def test_generate_report_handles_no_job_skills_detected():
    # If the job description has zero recognizable skills, coverage should
    # default to 1.0 rather than raising a ZeroDivisionError.
    report = generate_report("python developer", "we need someone great with people", doc_score=0.5, model=None)
    assert report["skill_coverage_pct"] == 100.0
