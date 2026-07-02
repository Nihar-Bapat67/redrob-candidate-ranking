"""
Stage 4 — rank.py  (THE INFERENCE ENGINE judges run)

Constraints honoured: CPU only, <=16GB, NO network, <=5 min. Loads only pre-built
artifacts; makes zero API calls.

Flow:
  1. load artifacts (FAISS index, candidate_ids, local mpnet model, LightGBM ranker)
  2. parse JD, embed a JD query, FAISS -> top 500 (recall)
  3. stream candidates.jsonl, keep only those 500 full records (memory-light)
  4. compute the 14-feature matrix for the 500 (precision)
  5. LightGBM scores; push Stage-2 honeypots below all clean candidates
  6. sort by (score desc, candidate_id asc), take top 100, write spec-valid CSV

Reasoning column here is a minimal factual one-liner; Stage 5 (reasoning.py)
replaces it with the full rank-aware templated reasoning.

Usage:  python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""

import os
import csv
import json
import time
import pickle
import argparse

import numpy as np

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.jd_parser import load_and_parse, build_jd_query
from pipeline.features import build_feature_row, FEATURE_COLUMNS, _load_coherence
from pipeline.reasoning import generate_reasoning

ARTIFACTS = "artifacts"
TOP_K_FAISS = 500
TOP_N_OUT = 100


def _load_full_records(path, wanted_ids):
    recs = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            cid = c.get("candidate_id")
            if cid in wanted_ids:
                recs[cid] = c
                if len(recs) == len(wanted_ids):
                    break
    return recs


def run(candidates="candidates.jsonl", jd_path="jd.txt", out="submission.csv",
        artifacts=ARTIFACTS, top_k=TOP_K_FAISS, coherence_path="coherence_scores.csv"):
    import faiss
    from sentence_transformers import SentenceTransformer
    import lightgbm as lgb

    t0 = time.time()
    jd = load_and_parse(jd_path)

    # --- load artifacts ---
    index = faiss.read_index(os.path.join(artifacts, "faiss.index"))
    ids = pickle.load(open(os.path.join(artifacts, "candidate_ids.pkl"), "rb"))
    model = SentenceTransformer(os.path.join(artifacts, "embedder"))  # local, no network
    booster = lgb.Booster(model_file=os.path.join(artifacts, "ranker.lgb"))
    feat_cols = json.load(open(os.path.join(artifacts, "feature_cols.json")))

    # --- retrieve top-K (recall) ---
    q = model.encode([build_jd_query(jd)], normalize_embeddings=True,
                     convert_to_numpy=True).astype("float32")
    k = min(top_k, len(ids))
    _, idx = index.search(q, k)
    cand_ids = [ids[i] for i in idx[0]]
    print(f"FAISS retrieved {len(cand_ids)} candidates in {time.time()-t0:.1f}s")

    # --- gather full records + Stage-2 coherence ---
    recs = _load_full_records(candidates, set(cand_ids))
    coherence = _load_coherence(coherence_path)

    # --- features + LightGBM scores (precision) ---
    feat_rows, valid = [], []
    for cid in cand_ids:
        c = recs.get(cid)
        if c is None:
            continue
        coh = coherence.get(cid)
        row = build_feature_row(c, jd, coh)
        feat_rows.append(row)
        valid.append(cid)

    import pandas as pd
    fdf = pd.DataFrame(feat_rows)
    fdf["coherence_score"] = fdf.get("coherence_score", 1.0)
    fdf["anomaly_score"] = fdf.get("anomaly_score", 0.0)
    fdf[["coherence_score", "anomaly_score"]] = fdf[["coherence_score", "anomaly_score"]].fillna(
        {"coherence_score": 1.0, "anomaly_score": 0.0})
    X = fdf[feat_cols].to_numpy(dtype=float)
    scores = booster.predict(X)

    # --- assemble; demote honeypots below all clean candidates ---
    results = []
    for i, cid in enumerate(valid):
        is_hp = fdf.iloc[i].get("rank_ceiling")
        is_hp = (is_hp is not None) and not (isinstance(is_hp, float) and np.isnan(is_hp))
        s = round(float(scores[i]), 6)
        results.append({"candidate_id": cid, "raw": s,
                        "score": (s - 1000.0) if is_hp else s,
                        "cand": recs[cid], "feats": fdf.iloc[i]})

    # round then sort: (score desc, candidate_id asc) -> spec tie-break
    for r in results:
        r["score"] = round(r["score"], 6)
    results.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    top = results[:TOP_N_OUT]

    # --- write spec-valid CSV ---
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, r in enumerate(top, start=1):
            w.writerow([r["candidate_id"], rank, f"{r['score']:.6f}",
                        generate_reasoning(rank, r["cand"], jd, r["feats"])])

    n_hp = sum(1 for r in top if r["score"] < -500)
    print(f"\nWrote {len(top)} rows -> {out}")
    print(f"Honeypots in top 100: {n_hp}")
    print(f"Total inference time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stage 4 inference: produce ranked submission CSV")
    ap.add_argument("--candidates", default="candidates.jsonl")
    ap.add_argument("--jd", default="jd.txt")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--artifacts", default=ARTIFACTS)
    ap.add_argument("--top-k", type=int, default=TOP_K_FAISS)
    args = ap.parse_args()
    run(args.candidates, args.jd, args.out, args.artifacts, args.top_k)
