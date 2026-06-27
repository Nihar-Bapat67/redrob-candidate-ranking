# Redrob AI — Intelligent Candidate Ranking

Ranks the top 100 candidates from `candidates.jsonl` for the released Senior AI
Engineer JD. The ranking step is **offline, CPU-only, no network, < 40 s**.

## TL;DR — reproduce the submission

```bash
# 1. install inference deps
pip install -r requirements.txt

# 2. produce the ranked CSV (this is the command judges run)
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

`rank.py` loads only pre-built artifacts from `artifacts/` and makes **zero API
calls**. If `artifacts/` is not present, build it once (see *Pre-computation*).

## Architecture (two phases)

This is a **knowledge-distillation** design: expensive LLM judgment is used
*offline* to teach a small, fast, local model that runs at inference.

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
| 1 Ingestion | `db_setup.py` | stream/parse/normalise 100k JSONL records |
| 2 Coherence | `coherence.py` | deterministic impossibility checks → honeypot `rank_ceiling` + soft anomaly signal |
| 3 Features | `canonicaliser.py`, `jd_parser.py`, `features.py` | 14 JD-grounded features (per-candidate) |
| 4 Ranking | `precompute.py`, `train_ranker.py`, `rank.py` | FAISS recall + LightGBM precision |
| 5 Reasoning | `reasoning.py` | rank-aware slot-filling (no runtime LLM → cannot hallucinate) |

## Pre-computation (build the artifacts)

Run all commands **from the project root** (scripts use root-relative paths and
import the shared modules at root):

```bash
pip install -r requirements-precompute.txt

python day2_coherence_validation/coherence.py        # -> coherence_scores.csv, coherence_ceiling.json
python day3_index_and_labels/precompute.py           # -> artifacts/faiss.index, candidate_ids.pkl, embedder/  (~45 min CPU)
python day3_index_and_labels/make_label_queue.py     # -> label_queue.jsonl (FAISS contenders + stratified)
GROQ_API_KEY=... python day4_labeling_and_training/label_llm.py --provider groq   # broad-net labels -> labels.jsonl (resumable)
python day4_labeling_and_training/train_ranker.py --honeypot-cap 70   # -> artifacts/ranker.lgb
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

## Sandbox / demo (spec §10.5)
A lightweight **Streamlit** app ranks a ≤100-candidate sample end-to-end (LightGBM
+ coherence + reasoning; no FAISS/embeddings needed at this scale). Run it locally:
```bash
pip install -r sandbox/requirements.txt
streamlit run sandbox/app.py
```
See [`sandbox/README.md`](sandbox/README.md) for one-click deploy to Hugging Face
Spaces / Streamlit Cloud / Docker.

## Repo map (organised day-wise; shared library + entry point kept at root)
```
# --- common (root): the entry point + shared library imported across stages ---
rank.py                 inference entry point (judges run this)
canonicaliser.py        skill alias + domain lexicons
jd_parser.py            JD -> structured spec (+ build_jd_query)
features.py             14-feature builder (per candidate)
reasoning.py            Stage 5 rank-aware reasoning
labels.jsonl            LLM relevance labels (training data)
jd.txt, requirements*.txt, submission.csv, submission_metadata.yaml

# --- day-wise development work ---
day1_data_ingestion/
    db_setup.py             streaming JSONL ingestion
day2_coherence_validation/
    coherence.py            honeypot/coherence validator (deterministic + soft anomaly)
day3_index_and_labels/
    precompute.py           offline FAISS build (MiniLM embeddings over 100k)
    make_label_queue.py     pick candidates to label (FAISS contenders + stratified)
day4_labeling_and_training/
    label_llm.py            offline LLM labeler (Groq/Gemini), resumable
    add_labels.py           merge label batches into labels.jsonl
    train_ranker.py         LightGBM lambdarank trainer
```
The day-folder scripts add the project root to `sys.path` so the shared modules
import cleanly when run from root.
