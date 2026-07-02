# Project Structure

Organised by **role in the system** (the way product teams lay out a service), not
by the day it was built. One code package, one entry point, clear separation of
code / data / artifacts / demos / docs.

```
redrob-candidate-ranking/
│
├── rank.py                     # ENTRY POINT — the command judges run:
│                               #   python rank.py --candidates ./candidates.jsonl --out ./submission.csv
│
├── src/pipeline/               # THE CODE — one importable package
│   ├── ingestion.py            #   Stage 1  · stream + normalise the 100k records
│   ├── coherence.py            #   Stage 2  · honeypot / coherence validator (rules + soft anomaly)
│   ├── canonicaliser.py        #   Stage 3a · skill aliases + domain lexicons
│   ├── jd_parser.py            #   Stage 3b · JD text -> structured spec (+ build_jd_query)
│   ├── features.py             #   Stage 3c · the 14 JD-grounded features
│   ├── precompute.py           #   Stage 4  · build the FAISS index (offline)
│   ├── make_label_queue.py     #   Stage 4  · choose candidates to label
│   ├── label_llm.py            #   Stage 4  · LLM labeler (Groq/Gemini, offline)
│   ├── add_labels.py           #   Stage 4  · merge label batches -> labels.jsonl
│   ├── train_ranker.py         #   Stage 4  · train the LightGBM ranker
│   └── reasoning.py            #   Stage 5  · rank-aware, fact-grounded reasons
│
├── artifacts/                  # MODEL + INDEX (loaded at inference)
│   ├── ranker.lgb              #   12 KB trained model            (committed)
│   ├── feature_cols.json       #   feature order                  (committed)
│   ├── faiss.index             #   100k vectors                   (git-ignored, rebuild via precompute.py)
│   ├── embeddings.npy          #   raw embeddings                 (git-ignored)
│   ├── candidate_ids.pkl       #   FAISS row -> candidate_id      (git-ignored)
│   └── embedder/               #   local MiniLM model             (git-ignored)
│
├── data / config (repo root)
│   ├── jd.txt                  #   the job description
│   ├── labels.jsonl            #   LLM/rule training labels (committed — reproduces the model)
│   ├── sample_candidates.json  #   120-candidate demo sample (yields a top-100)
│   ├── submission.csv          #   the final ranked top-100 output
│   ├── coherence_scores.csv    #   Stage-2 output used at inference (git-ignored)
│   ├── coherence_ceiling.json  #   honeypot ids (git-ignored)
│   ├── requirements.txt        #   inference deps
│   ├── requirements-precompute.txt
│   └── submission_metadata.yaml
│
├── sandbox/                    # DEMO 1 · Streamlit app (spec §10.5)
├── vercel-dashboard/           # DEMO 2 · web dashboard (static UI + serverless API)
├── docs/                       # the technical design document (md + docx)
├── README.md · LICENSE · .gitignore · .gitattributes
└── [PUB] India_runs…/          # organizers' spec bundle + validator (reference)
```

## Mental model
- **`src/pipeline/`** is the library. **`rank.py`** at the root is the single entry
  point that imports it and produces the submission.
- **Offline** scripts (`precompute`, `make_label_queue`, `label_llm`, `add_labels`,
  `train_ranker`) build the artifacts once; run them from the repo root, e.g.
  `python src/pipeline/train_ranker.py`.
- **`sandbox/`** and **`vercel-dashboard/`** are two front doors to the same engine.
- Big data / indexes / secrets are git-ignored and regenerable; the small model
  (`ranker.lgb`) and the labels are committed so the result reproduces from source.

## How a larger team would evolve this
Same shape, a few additions: turn `src/pipeline/` into an installed package
(`pyproject.toml`), add a `tests/` folder + CI, move data to cloud storage, and put
configs (thresholds, paths) into a `configs/` file instead of defaults in code.
