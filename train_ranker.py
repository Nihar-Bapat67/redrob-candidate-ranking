"""
Stage 4 (offline) — train_ranker.py

Trains the LightGBM learning-to-rank model that distills the LLM's recruiter
judgment into a small, fast, offline artifact.

Inputs:
  labels.jsonl            LLM relevance labels (0–5) from the approved rubric
  coherence_ceiling.json  Stage-2 honeypots -> added as gold relevance-0 rows
  candidates.jsonl        source records (features computed for labeled ids)
  coherence_scores.csv    Stage-2 coherence/anomaly features

Output:
  artifacts/ranker.lgb        trained LightGBM model
  artifacts/feature_cols.json feature column order (so rank.py matches exactly)

Single JD => one query group. LambdaRank optimises NDCG directly over that group.
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import lightgbm as lgb

from jd_parser import load_and_parse
from features import build_feature_row, FEATURE_COLUMNS, _load_coherence

ARTIFACTS = "artifacts"


def load_labels(path="labels.jsonl"):
    labels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                labels[r["candidate_id"]] = int(r["relevance"])
    return labels


def run(input_path="candidates.jsonl", labels_path="labels.jsonl",
        coherence_path="coherence_scores.csv", ceiling_path="coherence_ceiling.json",
        artifacts=ARTIFACTS, add_honeypots=True, honeypot_cap=30, val_frac=0.2, seed=42):
    jd = load_and_parse("jd.txt")
    labels = load_labels(labels_path)
    print(f"Loaded {len(labels)} LLM labels.")

    # Add a CAPPED sample of Stage-2 honeypots as gold relevance-0 so the model
    # also learns to push impossibles down — but only a small number, since the
    # inference rank-ceiling already removes them and a flood of 0s would drown
    # the real ranking signal (domain/seniority/availability).
    if add_honeypots and os.path.exists(ceiling_path):
        hp = [c for c in json.load(open(ceiling_path)) if c not in labels]
        rng0 = np.random.default_rng(seed)
        if len(hp) > honeypot_cap:
            hp = rng0.choice(hp, size=honeypot_cap, replace=False).tolist()
        for cid in hp:
            labels[cid] = 0
        print(f"Added {len(hp)} Stage-2 honeypots as gold relevance-0 (capped).")

    coherence = _load_coherence(coherence_path)
    wanted = set(labels)

    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            cid = c.get("candidate_id")
            if cid in wanted:
                row = build_feature_row(c, jd, coherence.get(cid))
                row["relevance"] = labels[cid]
                rows.append(row)
                if len(rows) == len(wanted):
                    break

    df = pd.DataFrame(rows)
    df["coherence_score"] = df.get("coherence_score", 1.0)
    df["anomaly_score"] = df.get("anomaly_score", 0.0)
    df[["coherence_score", "anomaly_score"]] = df[["coherence_score", "anomaly_score"]].fillna(
        {"coherence_score": 1.0, "anomaly_score": 0.0})
    print(f"Built features for {len(df)} labeled candidates.")
    print("Label distribution:", df["relevance"].value_counts().sort_index().to_dict())

    # train/val split (each a single query group for lambdarank)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    n_val = max(1, int(len(df) * val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["relevance"].to_numpy(dtype=int)

    dtrain = lgb.Dataset(X[tr_idx], label=y[tr_idx], group=[len(tr_idx)])
    dval = lgb.Dataset(X[val_idx], label=y[val_idx], group=[len(val_idx)], reference=dtrain)

    # Leaf/bin constraints scale with dataset size so small label sets can still
    # split (raise these as the label count grows).
    n_train = len(tr_idx)
    min_leaf = 5 if n_train < 200 else (10 if n_train < 1000 else 20)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10, 50],
        "boosting_type": "gbdt",
        "num_leaves": 15 if n_train < 300 else 31,
        "learning_rate": 0.05,
        "min_data_in_leaf": min_leaf,
        "min_data_in_bin": 1,
        "feature_fraction": 0.9,
        "bagging_fraction": 1.0,
        "label_gain": [0, 1, 3, 7, 15, 31],   # gains for relevance 0..5
        "verbose": -1,
    }
    print(f"\nTraining LightGBM (lambdarank), min_data_in_leaf={min_leaf}...")
    model = lgb.train(
        params, dtrain, num_boost_round=300,
        valid_sets=[dtrain, dval], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
    )

    os.makedirs(artifacts, exist_ok=True)
    model.save_model(os.path.join(artifacts, "ranker.lgb"))
    json.dump(FEATURE_COLUMNS, open(os.path.join(artifacts, "feature_cols.json"), "w"))
    print(f"\nSaved -> {os.path.join(artifacts, 'ranker.lgb')}")

    # feature importance (interview-defensible)
    imp = sorted(zip(FEATURE_COLUMNS, model.feature_importance(importance_type="gain")),
                 key=lambda kv: -kv[1])
    print("\nFeature importance (gain):")
    for name, g in imp:
        print(f"  {g:10.1f}  {name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="candidates.jsonl")
    ap.add_argument("--labels", default="labels.jsonl")
    ap.add_argument("--coherence", default="coherence_scores.csv")
    ap.add_argument("--ceiling", default="coherence_ceiling.json")
    ap.add_argument("--artifacts", default=ARTIFACTS)
    ap.add_argument("--no-honeypots", action="store_true")
    args = ap.parse_args()
    run(args.input, args.labels, args.coherence, args.ceiling, args.artifacts,
        add_honeypots=not args.no_honeypots)
