# 🎯 ResumeIQ — Explainable AI Resume × Job Match Scorer

> Scores how well your resume matches a job description, blends document-level
> semantic similarity with actual skill coverage, and shows you exactly which
> sentence in your resume triggered each skill match — even when you didn't
> use the job's exact wording.

Built with **Python**, **Sentence Transformers**, and **Streamlit**.

---

## What It Does

1. Upload your resume as a PDF
2. Paste any job description
3. Click "Analyze" → get:
   - A composite percentage match score (with a breakdown of *why*)
   - Skills you have that the job wants ✅, with evidence
   - Skills you're missing ❌
   - Bonus skills you have beyond what's required 💡
   - A plain-English recommendation

---

## How It Works

### 1 — PDF Text Extraction
`pdfplumber` extracts text from the resume, handling multi-column layouts better than basic PDF readers.

### 2 — Text Cleaning
Lowercased, special characters stripped, whitespace collapsed.

### 3 — Whole-Document Semantic Similarity
Both texts are embedded with the `all-MiniLM-L6-v2` Sentence Transformer (384-dim vectors) and compared with cosine similarity.

### 4 — Skill Detection (exact + semantic)
Each of 60+ predefined skills is checked two ways:
- **Exact match** — the literal phrase appears in the text (fast, 100% confidence).
- **Semantic match** — for skills not found exactly, we embed the skill phrase and every sentence in the resume, and count it as a match if the closest sentence clears a similarity threshold. This is what lets "developed machine learning systems" register as **machine learning** even without that literal phrase — and it's why every matched skill shows the evidence sentence and confidence score when you hover over it.

### 5 — Composite Score (not just similarity)
`score = 60% × document similarity + 40% × skill coverage`

A single whole-document similarity score can be gamed by a long, well-written but *irrelevant* resume. Blending in actual skill coverage (skills-in-common ÷ skills-the-job-wants) makes the score harder to game and easier to justify in an interview.

### 6 — Explainable Report
A recommendation is generated from the score band, and every skill pill is hoverable to show *why* it was matched.

---

## What Changed From v1 (and why it matters for an interview)

| v1 issue | Fix |
|---|---|
| Skills only matched on exact keyword — "built ML models" wouldn't count as "machine learning" | Added a semantic fallback pass using the same embedding model, with evidence + confidence shown per skill |
| Score was pure whole-document similarity, easy to game with a long irrelevant resume | Composite score = similarity + actual skill coverage, weighted and explained in-app |
| No tests | `pytest` suite covering text cleaning, exact skill detection, and score-bucket logic |
| No input limits | 5MB upload cap, minimum job description length, capped extracted text length |
| No logging | Basic logging on analysis events (no resume content logged, just sizes/scores) |

**Known limitations (worth saying out loud in an interview, not hiding):**
- Skill detection is still limited to a predefined list of ~60 skills — it won't catch a skill that's neither an exact match nor semantically close to anything on that list.
- No section-level weighting yet (a skill mentioned in the "Skills" header counts the same as one buried in an unrelated sentence). That's the next thing worth building.
- Scanned/image-only PDFs aren't supported (no OCR).

---

## Setup & Run (Command Prompt / Terminal)

**Prerequisites:** Python 3.10+, pip

```bash
cd resumeiq
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Browser opens automatically at `http://localhost:8501`. First run downloads the embedding model (~90MB); cached after that.

---

## How to Verify It Actually Works

1. Run the app (above) and upload any text-based PDF resume.
2. Paste a job description that clearly overlaps your resume (e.g. mentions "Python" if your resume says "Python").
3. Click **Analyze** — confirm the score, matched/missing/bonus skill columns, and the "How this score is calculated" expander all populate.
4. Hover over a matched skill pill — you should see an evidence sentence and confidence in the tooltip.
5. Try a job description that paraphrases a skill your resume has (e.g. job says "distributed systems," resume says "microservices architecture") and confirm it's still detected (marked with **≈**), demonstrating the semantic fallback.
6. Run the automated tests:
   ```bash
   pip install pytest  # already in requirements.txt
   pytest tests/ -v
   ```
   All 8 tests should pass — they run without downloading the embedding model, so they're fast.

---

## Project Structure

```
resumeiq/
├── app.py            ← Streamlit web UI
├── matcher.py         ← All AI/matching logic
├── requirements.txt
├── tests/
│   └── test_matcher.py
└── README.md
```

---

## Interview Talking Points

**"Tell me about this project."**
"ResumeIQ compares a resume against a job description using both whole-document semantic similarity and per-skill semantic matching — so it catches paraphrased skills, not just exact keyword hits. The score is a weighted composite of document similarity and skill coverage rather than similarity alone, specifically because pure similarity is easy to game with a long, generic resume. Every matched skill shows the evidence sentence and confidence, so the score is explainable, not a black box."

**"Why blend two scores instead of one?"**
"Document similarity alone rewards resumes that are topically similar even if they're missing hard requirements — a resume that talks a lot about 'software engineering' in general terms can score high against a job needing specific tools, even with zero of those tools mentioned. Skill coverage grounds the score in concrete requirements."

**"What would you improve next?"**
"Section-level weighting — a skill in a dedicated Skills section should count more than one mentioned in passing. I'd also expand skill detection past a fixed list, using NER or a fine-tuned classifier, and add OCR for scanned resumes."

---

## Technologies Used

| Tool | What It Does |
|---|---|
| `sentence-transformers` | Converts text to vectors for semantic comparison |
| `pdfplumber` | Extracts text from PDF resumes |
| `streamlit` | Web UI |
| `torch` | Deep learning engine under the hood |
| `pytest` | Automated tests |

---

## License

MIT License — free to use, modify, and share.
