"""
Redrob Candidate Ranking — Sandbox Demo (Streamlit)

Satisfies submission_spec Section 10.5: a hosted environment that accepts a small
candidate sample (<=100), runs the ranking system end-to-end, and produces a
ranked CSV — CPU-only, well under 5 minutes.

The sandbox ranks the PROVIDED sample directly: the 14-feature matrix + LightGBM
ranker + Stage-2 coherence (honeypot) demotion + Stage-5 reasoning. (FAISS is the
full system's recall layer over the 100k pool; for a <=100 sample we score them
all directly, exactly as the spec intends.)

Run locally:   streamlit run sandbox/app.py
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import lightgbm as lgb

# --- make the shared library + the Stage-2 validator importable from anywhere ---
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from pipeline.jd_parser import load_and_parse           # noqa: E402
from pipeline.features import build_feature_row, FEATURE_COLUMNS  # noqa: E402
from pipeline.reasoning import generate_reasoning        # noqa: E402
from pipeline.coherence import validate_coherence        # noqa: E402  (sklearn-free path)

ARTIFACTS = os.path.join(ROOT, "artifacts")


# --------------------------------------------------------------------------- #
# Core ranking (headless-testable — no Streamlit dependency)
# --------------------------------------------------------------------------- #
def load_model():
    booster = lgb.Booster(model_file=os.path.join(ARTIFACTS, "ranker.lgb"))
    feat_cols = json.load(open(os.path.join(ARTIFACTS, "feature_cols.json")))
    jd = load_and_parse(os.path.join(ROOT, "jd.txt"))
    return booster, feat_cols, jd


def rank_candidates(candidates, booster, feat_cols, jd):
    """Rank a list of candidate dicts -> spec-format DataFrame (candidate_id, rank, score, reasoning)."""
    rows, recs = [], []
    for c in candidates:
        coh = validate_coherence(c)   # deterministic coherence_score, rank_ceiling, flags
        feat_row = build_feature_row(c, jd, {
            "coherence_score": coh["coherence_score"],
            "anomaly_score": 0.0,     # population-level anomaly not available for an ad-hoc sample
            "rank_ceiling": coh["rank_ceiling"],
        })
        rows.append(feat_row)
        recs.append(c)

    fdf = pd.DataFrame(rows)
    fdf["coherence_score"] = fdf.get("coherence_score", 1.0)
    fdf["anomaly_score"] = fdf.get("anomaly_score", 0.0)
    X = fdf[feat_cols].to_numpy(dtype=float)
    scores = booster.predict(X)

    results = []
    for i, c in enumerate(recs):
        hp = fdf.iloc[i].get("rank_ceiling")
        is_hp = (hp is not None) and not (isinstance(hp, float) and np.isnan(hp))
        s = round(float(scores[i]), 6)
        results.append({
            "candidate_id": c.get("candidate_id"),
            "score": round(s - 1000.0 if is_hp else s, 6),
            "is_honeypot": is_hp,
            "cand": c, "feats": fdf.iloc[i],
        })

    # sort: score desc, candidate_id asc (spec tie-break)
    results.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    out = []
    for rank_i, r in enumerate(results, start=1):
        out.append({
            "candidate_id": r["candidate_id"],
            "rank": rank_i,
            "score": f"{r['score']:.6f}",
            "reasoning": generate_reasoning(rank_i, r["cand"], jd, r["feats"]),
        })
    return pd.DataFrame(out)


def parse_upload(raw_bytes):
    """Accept a .jsonl (one JSON per line) or a .json (list) of candidates."""
    text = raw_bytes.decode("utf-8", "ignore").strip()
    cands = []
    try:                                  # JSON array?
        data = json.loads(text)
        cands = data if isinstance(data, list) else [data]
    except json.JSONDecodeError:          # JSONL
        for line in text.splitlines():
            line = line.strip()
            if line:
                cands.append(json.loads(line))
    return [c for c in cands if c.get("candidate_id")]


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #
def main():
    import streamlit as st

    st.set_page_config(page_title="Redrob Candidate Ranker", page_icon="🎯", layout="wide")
    st.title("🎯 Redrob — AI Candidate Ranking (Sandbox)")
    st.caption("Senior AI Engineer JD · LightGBM ranker distilled from LLM relevance "
               "labels · CPU-only · no network at ranking.")

    with st.expander("How this works", expanded=False):
        st.markdown(
            "- Upload up to **100 candidates** (`.jsonl` or `.json`) or use the bundled sample.\n"
            "- Each candidate gets the **14 JD-grounded features**; a **LightGBM** model "
            "(trained offline on Claude+Groq relevance labels) scores them.\n"
            "- **Stage-2 coherence** flags impossible/honeypot profiles and demotes them.\n"
            "- **Stage-5** produces a rank-aware, fact-grounded reasoning line per candidate.\n"
            "- The full system first uses **FAISS** to retrieve the top 500 from the 100k pool; "
            "this sandbox ranks your sample directly (spec §10.5)."
        )

    booster, feat_cols, jd = load_model()

    src = st.radio("Input", ["Use bundled 50-candidate sample", "Upload my own (≤100)"],
                   horizontal=True)
    candidates = []
    if src.startswith("Use bundled"):
        sample = json.load(open(os.path.join(ROOT, "sample_candidates.json")))
        candidates = sample if isinstance(sample, list) else [sample]
        st.info(f"Loaded {len(candidates)} bundled sample candidates.")
    else:
        up = st.file_uploader("Candidates file (.jsonl or .json)", type=["jsonl", "json"])
        if up is not None:
            candidates = parse_upload(up.read())
            st.info(f"Parsed {len(candidates)} candidates.")

    if candidates:
        if len(candidates) > 100:
            st.warning(f"Sandbox caps at 100; using the first 100 of {len(candidates)}.")
            candidates = candidates[:100]

        import time
        t0 = time.time()
        df = rank_candidates(candidates, booster, feat_cols, jd)
        st.success(f"Ranked {len(df)} candidates in {time.time()-t0:.2f}s "
                   f"(honeypots demoted to the bottom).")

        st.dataframe(df, use_container_width=True, height=460)
        st.download_button("⬇️ Download ranked CSV", df.to_csv(index=False).encode(),
                           file_name="ranked_submission.csv", mime="text/csv")


if __name__ == "__main__":
    main()
