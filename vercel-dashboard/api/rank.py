"""
Vercel Python serverless function: POST /api/rank

Body: { "jd_text": "<job description text>", "candidates": [ <candidate dicts> ] }
Returns: { stages: {...}, ranked: [...], csv: "..." }

Runs the real ranking on the provided candidate sample: Stage-2 coherence
(honeypot) demotion -> 14-feature matrix -> LightGBM -> Stage-5 reasoning.
(The full FAISS recall over 100k candidates is the OFFLINE step; at web-sample
scale we score every provided candidate directly.)
"""

import os
import sys
import json
import io
import csv as csvmod
from http.server import BaseHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "_pipeline"))
ART = os.path.join(HERE, "artifacts")
MAX_CANDIDATES = 500
TOP_N_OUT = 100          # the submission requires a ranked top-100

_BOOSTER = None
_COLS = None


def _load():
    global _BOOSTER, _COLS
    if _BOOSTER is None:
        import lightgbm as lgb
        _BOOSTER = lgb.Booster(model_file=os.path.join(ART, "ranker.lgb"))
        _COLS = json.load(open(os.path.join(ART, "feature_cols.json")))
    return _BOOSTER, _COLS


def run_ranking(body):
    import numpy as np
    from jd_parser import parse_jd, load_and_parse
    from features import build_feature_row
    from reasoning import generate_reasoning
    from coherence import validate_coherence

    booster, feat_cols = _load()

    jd_text = (body.get("jd_text") or "").strip()
    jd = parse_jd(jd_text) if len(jd_text) > 60 else load_and_parse(os.path.join(HERE, "jd.txt"))

    cands = [c for c in (body.get("candidates") or []) if c.get("candidate_id")][:MAX_CANDIDATES]

    rows, recs, n_hp = [], [], 0
    for c in cands:
        coh = validate_coherence(c)
        if coh["rank_ceiling"] is not None:
            n_hp += 1
        rows.append(build_feature_row(c, jd, {
            "coherence_score": coh["coherence_score"],
            "anomaly_score": 0.0,
            "rank_ceiling": coh["rank_ceiling"],
        }))
        recs.append(c)

    if rows:
        X = np.array([[float(r.get(col) if r.get(col) is not None else 0.0)
                       for col in feat_cols] for r in rows], dtype=float)
        scores = booster.predict(X)
    else:
        scores = []

    results = []
    for i, c in enumerate(recs):
        hp = rows[i].get("rank_ceiling") is not None
        s = round(float(scores[i]), 6)
        results.append({"candidate_id": c["candidate_id"],
                        "score": round(s - 1000.0 if hp else s, 6),
                        "cand": c, "feats": rows[i]})
    results.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    results = results[:TOP_N_OUT]          # keep the ranked top 100

    ranked = []
    for rank_i, r in enumerate(results, start=1):
        p = r["cand"].get("profile") or {}
        ranked.append({
            "candidate_id": r["candidate_id"],
            "rank": rank_i,
            "score": f"{r['score']:.6f}",
            "title": p.get("current_title"),
            "years": p.get("years_of_experience"),
            "reasoning": generate_reasoning(rank_i, r["cand"], jd, r["feats"]),
        })

    buf = io.StringIO()
    w = csvmod.writer(buf)
    w.writerow(["candidate_id", "rank", "score", "reasoning"])
    for r in ranked:
        w.writerow([r["candidate_id"], r["rank"], r["score"], r["reasoning"]])

    return {
        "stages": {"total": len(cands), "honeypots": n_hp, "ranked": len(ranked),
                   "required_skills": len(jd.get("required_skills", []))},
        "ranked": ranked,
        "csv": buf.getvalue(),
    }


class handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "content-type")

    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()

    def do_POST(self):
        try:
            n = int(self.headers.get("content-length", 0) or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            result = run_ranking(body)
            out = json.dumps(result).encode("utf-8")
            self.send_response(200); self._cors()
            self.send_header("content-type", "application/json"); self.end_headers()
            self.wfile.write(out)
        except Exception as e:
            err = json.dumps({"error": str(e)}).encode("utf-8")
            self.send_response(500); self._cors()
            self.send_header("content-type", "application/json"); self.end_headers()
            self.wfile.write(err)
