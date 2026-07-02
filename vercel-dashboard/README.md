# Redrob Ranking — Vercel Dashboard

An interactive ML dashboard (static frontend + Python serverless backend) that runs
the real ranking pipeline on an uploaded candidate sample and visualises the stages.

```
vercel-dashboard/            <-- set this as the Vercel "Root Directory"
  public/
    index.html               frontend: PDF JD upload, candidates upload, animated
                             4-stage pipeline, results table + CSV download
    sample_candidates.json   120-candidate demo set ("Use sample data") -> ranked top-100
  api/
    rank.py                  serverless function  ->  POST /api/rank
    requirements.txt         lightgbm, numpy, pandas
    _pipeline/               vendored ranking modules (jd_parser, features,
                             reasoning, coherence, canonicaliser)
    artifacts/               ranker.lgb (12 KB) + feature_cols.json
    jd.txt                   fallback JD if none is uploaded
  vercel.json                function memory/duration
```

## Deploy to Vercel
1. Push this repo to GitHub (already done).
2. vercel.com → **Add New… → Project** → import `redrob-candidate-ranking`.
3. **Root Directory → `vercel-dashboard`**.  Framework Preset: **Other**.
4. **Deploy.** Vercel serves `public/` statically and turns `api/rank.py` into a
   Python serverless function (installs `api/requirements.txt` automatically).
5. Open the URL → upload a JD PDF + a candidates `.jsonl/.json` (or click
   *Use sample data*) → **Proceed to Rank**.

### Notes / limits
- **Why it ranks the uploaded sample (not 100k):** Vercel serverless caps at ~250 MB
  and ≤60 s with no GPU, so the FAISS+`sentence-transformers`+`torch` recall layer
  (~2 GB) can't run there. It's an offline step. The web app runs the rest of the
  pipeline (honeypot removal → 14-feature matrix → LightGBM → reasoning) directly on
  your sample — fast and faithful. Upload ≤ 500 candidates for the live demo.
- `maxDuration: 60` needs a **Pro** plan; on Hobby it is capped at 10 s (fine for warm
  calls; a cold start may occasionally need a retry).
- The JD PDF is parsed **in the browser** (pdf.js) and only the extracted text is sent.

## Run locally (optional)
```bash
# backend (from vercel-dashboard/api)
python -c "from http.server import HTTPServer; from rank import handler; HTTPServer(('127.0.0.1',8899),handler).serve_forever()"
# serve the frontend (from vercel-dashboard/public) and point fetch at the local API
python -m http.server 3000
```
