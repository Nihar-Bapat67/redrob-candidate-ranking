"""
Stage 4 (offline) — make_label_queue.py

Selects WHICH candidates to send to the LLM labeller and writes compact, faithful
payloads. Labeling is most valuable where it changes the ranking, so the queue is
ordered:

  1. FAISS top contenders for the JD  (the candidates that will actually compete)
  2. a stratified spread across domain_fit bands  (so the model also learns the
     mid/low range and calibrates the full 0–5 scale)

Honeypots flagged by Stage 2 are EXCLUDED here — they are added later as gold
relevance-0 rows in train_ranker.py (deterministic, no LLM tokens wasted on math).

Output: label_queue.jsonl — one compact JSON payload per line, contenders first.
"""

import os
import json
import pickle
import argparse

import numpy as np

# allow importing the shared library modules kept at the project root
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jd_parser import load_and_parse, build_jd_query
from features import domain_fit

ARTIFACTS = "artifacts"
DESC_CHARS = 280


def compact_payload(c):
    p = c.get("profile") or {}
    career = []
    for j in (c.get("career_history") or [])[:6]:
        career.append({
            "title": j.get("title"),
            "company": j.get("company"),
            "industry": j.get("industry"),
            "months": j.get("duration_months"),
            "desc": (j.get("description") or "")[:DESC_CHARS],
        })
    edu = [{"degree": e.get("degree"), "field": e.get("field_of_study"),
            "end_year": e.get("end_year")} for e in (c.get("education") or [])[:3]]
    s = c.get("redrob_signals") or {}
    return {
        "candidate_id": c.get("candidate_id"),
        "current_title": p.get("current_title"),
        "years_experience": p.get("years_of_experience"),
        "career": career,
        "education": edu,
        "skills": [sk.get("name") for sk in (c.get("skills") or [])][:20],
        "signals": {
            "last_active": s.get("last_active_date"),
            "recruiter_response_rate": s.get("recruiter_response_rate"),
            "open_to_work": s.get("open_to_work_flag"),
            "interview_completion": s.get("interview_completion_rate"),
        },
    }


def run(input_path="candidates.jsonl", artifacts=ARTIFACTS,
        n_contenders=1500, n_sample=1500, out="label_queue.jsonl",
        ceiling_path="coherence_ceiling.json"):
    import faiss
    from sentence_transformers import SentenceTransformer

    jd = load_and_parse("jd.txt")

    # honeypots to exclude from LLM labeling
    honeypots = set()
    if os.path.exists(ceiling_path):
        honeypots = set(json.load(open(ceiling_path)))
    print(f"Excluding {len(honeypots)} Stage-2 honeypots from LLM queue.")

    # --- FAISS contenders ---
    index = faiss.read_index(os.path.join(artifacts, "faiss.index"))
    ids = pickle.load(open(os.path.join(artifacts, "candidate_ids.pkl"), "rb"))
    model = SentenceTransformer(os.path.join(artifacts, "embedder"))
    q = model.encode([build_jd_query(jd)], normalize_embeddings=True,
                     convert_to_numpy=True).astype("float32")
    k = min(n_contenders * 2, len(ids))
    _, idx = index.search(q, k)
    contender_ids = []
    for i in idx[0]:
        cid = ids[i]
        if cid not in honeypots:
            contender_ids.append(cid)
        if len(contender_ids) >= n_contenders:
            break
    contender_set = set(contender_ids)
    print(f"Selected {len(contender_ids)} FAISS contenders.")

    # --- stratified spread by domain_fit over the rest ---
    rng = np.random.default_rng(42)
    bands = {b: [] for b in range(5)}  # 5 bands: [0,.2),...,[.8,1]
    full = {}
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            cid = c.get("candidate_id")
            if not cid or cid in honeypots:
                continue
            full[cid] = c
            if cid in contender_set:
                continue
            df = domain_fit(c, jd)
            b = min(4, int(df * 5))
            bands[b].append(cid)

    per_band = max(1, n_sample // 5)
    sample_ids = []
    for b in range(5):
        pool = bands[b]
        take = min(per_band, len(pool))
        sample_ids.extend(rng.choice(pool, size=take, replace=False).tolist()
                          if pool else [])
    print("Stratified sample by domain_fit band:",
          {b: min(per_band, len(bands[b])) for b in range(5)})

    # --- write queue (contenders first) ---
    queue_ids = contender_ids + [c for c in sample_ids if c not in contender_set]
    with open(out, "w", encoding="utf-8") as w:
        for cid in queue_ids:
            w.write(json.dumps(compact_payload(full[cid]), ensure_ascii=False) + "\n")
    print(f"\nWrote {len(queue_ids)} payloads -> {out} (contenders first)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="candidates.jsonl")
    ap.add_argument("--artifacts", default=ARTIFACTS)
    ap.add_argument("--n-contenders", type=int, default=1500)
    ap.add_argument("--n-sample", type=int, default=1500)
    ap.add_argument("--out", default="label_queue.jsonl")
    args = ap.parse_args()
    run(args.input, args.artifacts, args.n_contenders, args.n_sample, args.out)
