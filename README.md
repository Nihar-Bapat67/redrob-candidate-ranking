# Redrob AI — Intelligent Candidate Ranking

Ranks the top 100 candidates from `candidates.jsonl` for the released Senior AI
Engineer JD. The ranking step is **offline, CPU-only, no network.

## TL;DR — reproduce the submission

```bash
# 0. clone
git clone https://github.com/Nihar-Bapat67/redrob-candidate-ranking
cd redrob-candidate-ranking

# 1. fetch the large FAISS artifacts (too big for git) from the GitHub Release,
#    then extract in place. This creates artifacts/faiss.index, artifacts/embedder/,
#    artifacts/candidate_ids.pkl and coherence_scores.csv.
#    Use curl.exe -L -o  (writes straight to disk, works the same on Windows/Linux/macOS):
curl.exe -L -o artifacts.tar.gz https://github.com/Nihar-Bapat67/redrob-candidate-ranking/releases/download/artifacts-v1.0/artifacts.tar.gz
tar -xzf artifacts.tar.gz          # bundled tar works in PowerShell, Git Bash, Linux, macOS

# 2. install inference deps
pip install -r requirements.txt

# 3. produce the ranked CSV (this is the command judges run)
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

> ⚠️ **Windows / PowerShell — do NOT use bare `wget`.** In PowerShell `wget` is an
> alias for `Invoke-WebRequest`, which (without `-OutFile`) downloads the file into
> memory and prints a response object **instead of writing `artifacts.tar.gz` to
> disk** — `tar` then fails with `Error opening archive: Failed to open
> 'artifacts.tar.gz'` and `rank.py` crashes with `could not open
> artifacts\faiss.index`. Use **`curl.exe -L -o artifacts.tar.gz <url>`** (the `.exe`
> matters — it forces the real curl, not the PowerShell alias). If you must use
> `Invoke-WebRequest`, add `-OutFile artifacts.tar.gz`.
>
> Verify the download before extracting — it should be ~218 MB:
> ```powershell
> (Get-Item artifacts.tar.gz).Length    # ~227864805 bytes
> ```

### Why a separate download?

The FAISS recall layer needs three large, git-ignored artifacts that exceed
GitHub's file limits, so they ship as a versioned **GitHub Release** asset
(`artifacts.tar.gz`, ~218 MB) instead of living in the repo:

| File (extracts to) | Size | Needed at inference |
|--------------------|-----:|:-------------------:|
| `artifacts/faiss.index`         | 147 MB | ✅ FAISS recall over the 100k pool |
| `artifacts/embedder/`           |  88 MB | ✅ local MiniLM (embeds the JD query, no network) |
| `artifacts/candidate_ids.pkl`   | 1.5 MB | ✅ FAISS row → candidate_id map |
| `coherence_scores.csv`          | 6.1 MB | ✅ Stage-2 honeypot / coherence signal |

The small model itself (`artifacts/ranker.lgb`, `artifacts/feature_cols.json`) **is**
committed to the repo. After extracting the release, `rank.py` loads everything from
disk and makes **zero API calls**.

> **No curl?** Download the asset from the
> [Releases page](https://github.com/Nihar-Bapat67/redrob-candidate-ranking/releases)
> in a browser, drop `artifacts.tar.gz` in the repo root, then `tar -xzf artifacts.tar.gz`.
>
> **Prefer to rebuild instead of download?** Run `python src/pipeline/precompute.py`
> (~45 min, CPU) to regenerate the exact same artifacts from source.

## Architecture (two phases)

This is a **knowledge-distillation** design: expensive LLM judgment is used
*offline* to teach a small, fast, local model that runs at inference.

### Repository structure

- Core runtime entry points stay at the repository root: [rank.py](rank.py), [submission.csv](submission.csv), [submission_metadata.yaml](submission_metadata.yaml)
- Shared pipeline modules now live under [src/pipeline](src/pipeline)
- Documentation and architecture notes live under [docs/architecture](docs/architecture)
- The sandbox demo remains in [sandbox/app.py](sandbox/app.py)

### Daily workflow

1. Run ingestion/coherence/feature steps from the pipeline package.
2. Build offline artifacts with the precompute and training scripts.
3. Generate the final submission with [rank.py](rank.py).

**Offline pre-computation** (no time limit, network allowed):
1. **Embeddings → FAISS** (`precompute.py`): embed all 100k profiles with
   `all-MiniLM-L6-v2` into a FAISS `IndexFlatIP` (the *recall* layer).
2. **Relevance labels** (`label_llm.py` + in-session Claude): LLMs score
   candidate-JD relevance 0–5 using a fixed recruiter rubric → `labels.jsonl`.
3. **Train ranker** (`train_ranker.py`): LightGBM `lambdarank` on 14 hand-built,
   JD-grounded features → `artifacts/ranker.lgb`.

**Inference** (`rank.py`, ≤ 5 min, CPU, 16 GB, no network):
JD → FAISS top-500 (recall) → 14 features → LightGBM re-rank (precision) →
demote honeypots → top-100 → spec-valid CSV with rank-aware reasoning.

### Why it beats keyword matching
The dataset is a deliberate **keyword trap**: the `skills` array is assigned
~uniformly at random, so an "HR Manager" can list FAISS + RAG. So skill overlap is
near-useless (mean `skill_match_pct` ≈ 0.02). The decisive feature, **`domain_fit`**,
is built from the candidate's **titles + career descriptions**, not their skill
list — an HR Manager scores ≈ 0.09 while an ML Engineer scores ≈ 0.90.

### The 5 stages
| Stage | File(s) | Role |
|------|---------|------|
| 1 Ingestion | `ingestion.py` | stream/parse/normalise 100k JSONL records |
| 2 Coherence | `coherence.py` | deterministic impossibility checks → honeypot `rank_ceiling` + soft anomaly signal |
| 3 Features | `canonicaliser.py`, `jd_parser.py`, `features.py` | 14 JD-grounded features (per-candidate) |
| 4 Ranking | `precompute.py`, `train_ranker.py`, `rank.py` | FAISS recall + LightGBM precision |
| 5 Reasoning | `reasoning.py` | rank-aware slot-filling (no runtime LLM → cannot hallucinate) |

## Pre-computation (build the artifacts)

Run all commands **from the project root** (scripts use root-relative paths and
import the shared modules at root):

```bash
pip install -r requirements-precompute.txt

python src/pipeline/coherence.py        # -> coherence_scores.csv, coherence_ceiling.json
python src/pipeline/precompute.py           # -> artifacts/faiss.index, candidate_ids.pkl, embedder/  (~45 min CPU)
python src/pipeline/make_label_queue.py     # -> label_queue.jsonl (FAISS contenders + stratified)
GROQ_API_KEY=... python src/pipeline/label_llm.py --provider groq   # broad-net labels -> labels.jsonl (resumable)
python src/pipeline/train_ranker.py --honeypot-cap 70   # -> artifacts/ranker.lgb
```

Large artifacts (`faiss.index`, `embeddings.npy`, `embedder/`) are git-ignored
(too big for GitHub); regenerate them with `precompute.py`. The Claude/LLM
relevance labels (`labels.jsonl`) and the trained `ranker.lgb` are small and the
labels are committed so training is reproducible.

## Compute constraints satisfied
- **Runtime:** `rank.py` ≈ 30–40 s (limit 5 min).
- **Memory:** streams the file, keeps only the 500 retrieved records.
- **CPU only / no GPU:** `faiss-cpu`, CPU torch, LightGBM.
- **No network at ranking:** model + index loaded from disk; LLMs used *only* offline.

## Sandbox 
A lightweight **Streamlit** app ranks a ≤100-candidate sample end-to-end (LightGBM
+ coherence + reasoning; no FAISS/embeddings needed at this scale). Run it locally:
```bash
pip install -r sandbox/requirements.txt
streamlit run sandbox/app.py
```
See [`sandbox/README.md`](sandbox/README.md) for one-click deploy to Hugging Face
Spaces / Streamlit Cloud / Docker.

## Repo map (shared pipeline package + root entry points)
```
# --- common (root): the entry point + shared runtime files ---
rank.py                 inference entry point (judges run this)
submission.csv          ranked submission output
submission_metadata.yaml submission metadata
jd.txt, requirements*.txt, labels.jsonl

# --- pipeline package ---
src/pipeline/
    coherence.py         honeypot/coherence validator (deterministic + soft anomaly)
    precompute.py        offline FAISS build (MiniLM embeddings over 100k)
    make_label_queue.py  pick candidates to label (FAISS contenders + stratified)
    label_llm.py         offline LLM labeler (Groq/Gemini), resumable
    add_labels.py        merge label batches into labels.jsonl
    train_ranker.py      LightGBM trainer
    canonicaliser.py     skill alias + domain lexicons
    jd_parser.py         JD -> structured spec (+ build_jd_query)
    features.py          14-feature builder (per candidate)
    reasoning.py         Stage 5 rank-aware reasoning
    ingestion.py         streaming JSONL ingestion helper
```
The pipeline scripts are now called from the package path directly, keeping the
repository structure easy to navigate and consistent with a product team workflow.
