"""
Stage 4 (offline) — precompute.py

Builds the FAISS retrieval artifacts ONCE, offline, before the 5-minute clock.
Network/time are unrestricted here; only the inference step (rank.py) is constrained.

Produces in artifacts/:
  faiss.index        IndexFlatIP over L2-normalised 768-d embeddings (cosine sim)
  candidate_ids.pkl  parallel list mapping FAISS row -> candidate_id
  embeddings.npy     the raw normalised embedding matrix (reused for labeling/EDA)
  mpnet/             the sentence-transformers model saved locally so rank.py can
                     load it with local_files_only=True (no network at inference)

The profile text blob deliberately concatenates the SEMANTIC signal — title,
headline, summary, and career titles+descriptions — because in this dataset the
skills list is random noise; the descriptions are the truth-teller.
"""

import os
import json
import pickle
import argparse
import multiprocessing as mp

import numpy as np

ARTIFACTS = "artifacts"
# MiniLM-L6-v2: 5x smaller/faster than mpnet on CPU. FAISS is only the recall
# layer (LightGBM re-ranks the top 500), so this is the right speed/quality call.
MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_BATCH = 256
MAX_BLOB_CHARS = 600   # shorter blob -> much faster embedding; keeps the recall signal


def build_blob(candidate):
    """Assemble the text used to embed a candidate (semantic fields only)."""
    p = candidate.get("profile") or {}
    parts = [
        p.get("current_title") or "",
        p.get("headline") or "",
        (p.get("summary") or "")[:400],
    ]
    # career TITLES + companies carry domain signal cheaply (skip long descriptions)
    for job in (candidate.get("career_history") or []):
        parts.append(job.get("title") or "")
        parts.append(job.get("company") or "")
    blob = ". ".join(s for s in parts if s).strip()
    return blob[:MAX_BLOB_CHARS]


def iter_candidates(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("candidate_id"):
                yield d


def run(input_path="candidates.jsonl", artifacts=ARTIFACTS, limit=None):
    import torch
    torch.set_num_threads(mp.cpu_count())
    import faiss
    from sentence_transformers import SentenceTransformer

    os.makedirs(artifacts, exist_ok=True)
    print(f"Loading embedding model '{MODEL_NAME}' on {mp.cpu_count()} threads...",
          flush=True)
    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()
    print(f"  embedding dim = {dim}", flush=True)

    # save model locally for offline inference
    model_dir = os.path.join(artifacts, "embedder")
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        print(f"  saving model -> {model_dir}")
        model.save(model_dir)

    ids, blobs = [], []
    print(f"Reading candidates from {input_path}...")
    for d in iter_candidates(input_path):
        ids.append(d["candidate_id"])
        blobs.append(build_blob(d))
        if limit and len(ids) >= limit:
            break
    n = len(ids)
    print(f"  {n} candidates to embed")

    # embed in batches, L2-normalise for inner-product = cosine
    embs = np.zeros((n, dim), dtype="float32")
    for start in range(0, n, EMBED_BATCH):
        end = min(start + EMBED_BATCH, n)
        vecs = model.encode(
            blobs[start:end],
            batch_size=EMBED_BATCH,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embs[start:end] = vecs.astype("float32")
        if (end // EMBED_BATCH) % 20 == 0 or end == n:
            print(f"  embedded {end}/{n}", flush=True)

    print("Building FAISS IndexFlatIP...")
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    print(f"  index.ntotal = {index.ntotal}")

    faiss.write_index(index, os.path.join(artifacts, "faiss.index"))
    with open(os.path.join(artifacts, "candidate_ids.pkl"), "wb") as f:
        pickle.dump(ids, f)
    np.save(os.path.join(artifacts, "embeddings.npy"), embs)

    print("\nSaved artifacts:", flush=True)
    for fn in ("faiss.index", "candidate_ids.pkl", "embeddings.npy", "embedder/"):
        print(f"  {os.path.join(artifacts, fn)}")
    print("DONE.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build FAISS retrieval artifacts (offline)")
    ap.add_argument("--input", default="candidates.jsonl")
    ap.add_argument("--artifacts", default=ARTIFACTS)
    ap.add_argument("--limit", type=int, default=None, help="embed only first N (dev)")
    args = ap.parse_args()
    run(args.input, args.artifacts, args.limit)
