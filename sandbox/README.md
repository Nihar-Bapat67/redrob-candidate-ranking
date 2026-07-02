# Redrob Candidate Ranker — Sandbox Demo

The hosted demo required by **submission_spec §10.5**: accepts a candidate sample,
runs the ranking system **end-to-end**, and produces a ranked **top-100** CSV —
CPU-only, well under 5 minutes.

## What it does
- Upload candidates (`.jsonl` or `.json`) **or** use the bundled 120-candidate sample;
  the app scores **all** of them and returns the ranked **top 100** (feed it ≥100 to
  fill a complete top-100).
- Computes the **14 JD-grounded features**, scores with the **LightGBM** ranker
  (`artifacts/ranker.lgb`, trained offline on Claude+Groq relevance labels),
  applies **Stage-2 coherence** honeypot demotion, and writes **Stage-5** rank-aware
  reasoning.
- Shows the ranked table and a **Download ranked CSV** button.

> The full system uses **FAISS** to retrieve the top 500 from the 100k pool first;
> a ≤100 sample is small enough to rank directly, which is exactly what §10.5 checks.
> So this demo needs **no embeddings/FAISS/torch** — only the 12 KB LightGBM model.

## Run locally
```bash
pip install -r sandbox/requirements.txt
streamlit run sandbox/app.py
# open http://localhost:8501
```

## Deploy (pick one)

### A) Hugging Face Spaces  (recommended — free, controls its own deps)
1. Create a new **Space** → SDK: **Streamlit**.
2. In the Space, add the repo files (push, or "Sync with GitHub").
3. Put this at the **top of the Space's `README.md`** so HF knows the entry point:
   ```
   ---
   title: Redrob Candidate Ranker
   emoji: 🎯
   sdk: streamlit
   app_file: sandbox/app.py
   pinned: false
   ---
   ```
4. Set the Space's `requirements.txt` to the contents of `sandbox/requirements.txt`
   (lean — avoids installing torch/faiss).
5. The Space builds and serves the app. Paste its URL into `submission_metadata.yaml → sandbox_link`.

### B) Streamlit Community Cloud
1. Connect this GitHub repo.
2. **Main file path:** `sandbox/app.py`.
3. **Advanced → Python deps:** point at `sandbox/requirements.txt` (or copy its lines),
   so it doesn't try to install the heavy inference deps from the root `requirements.txt`.

### C) Docker (self-contained `docker run` recipe)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r sandbox/requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "sandbox/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
```bash
docker build -t redrob-sandbox . && docker run -p 8501:8501 redrob-sandbox
```

## Files the sandbox depends on (already in the repo)
`artifacts/ranker.lgb`, `artifacts/feature_cols.json`, `jd.txt`,
`sample_candidates.json`, and the shared modules in `src/pipeline`.
