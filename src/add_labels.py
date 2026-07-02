"""
Stage 4 (offline) — add_labels.py

Merges a batch of LLM labels into the persistent label store labels.jsonl.
Resumable: dedups by candidate_id (last write wins). Input is a JSON array of
objects with at least {"candidate_id", "relevance"} (an optional "reasoning" is
kept for audit).

Usage:  python add_labels.py _batch_labels.json
"""

import sys
import json

STORE = "labels.jsonl"


def load_store():
    out = {}
    try:
        with open(STORE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    out[r["candidate_id"]] = r
    except FileNotFoundError:
        pass
    return out


def main():
    if len(sys.argv) != 2:
        print("Usage: python add_labels.py <batch.json>")
        sys.exit(1)

    batch = json.load(open(sys.argv[1], "r", encoding="utf-8"))
    store = load_store()

    added, updated, bad = 0, 0, 0
    for r in batch:
        cid = r.get("candidate_id")
        rel = r.get("relevance")
        if not cid or rel is None:
            bad += 1
            continue
        rel = int(rel)
        if not 0 <= rel <= 5:
            bad += 1
            continue
        rec = {"candidate_id": cid, "relevance": rel}
        if r.get("reasoning"):
            rec["reasoning"] = r["reasoning"]
        if cid in store:
            updated += 1
        else:
            added += 1
        store[cid] = rec

    with open(STORE, "w", encoding="utf-8") as w:
        for rec in store.values():
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"added={added} updated={updated} skipped_bad={bad} total={len(store)}")


if __name__ == "__main__":
    main()
