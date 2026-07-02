"""
Stage 5 — reasoning.py

Generates the `reasoning` column by SLOT-FILLING from facts extracted literally
from each candidate's profile — never an LLM at inference (no network, and
structurally impossible to hallucinate).

Design: instead of a few rigid whole-sentence templates, each reason is ASSEMBLED
from grounded clauses —
  head    : title (years) at company
  + up to 3 real STRENGTHS, each gated by a feature threshold
  + the single most salient real CONCERN (honest)
  + a rank-tracking VERDICT phrase
This yields combinatorial variety across the top 100 (satisfying the Stage-4
"substantively different" check) while every clause is tied to a real value, and
the verdict strictly tracks rank (satisfying the "rank-consistency" check).
"""

import sys
from datetime import datetime
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from .canonicaliser import canonicalise_set
except ImportError:  # pragma: no cover - direct script execution fallback
    from canonicaliser import canonicalise_set

CURRENT_DATE = datetime(2026, 6, 1)

# skill buckets used only to *name* the sub-domain in the strength clause
_RETRIEVAL = {"FAISS", "Pinecone", "Weaviate", "Qdrant", "Milvus", "OpenSearch",
              "Elasticsearch", "pgvector", "Vector Search", "Semantic Search",
              "Information Retrieval", "BM25", "Ranking Systems", "Learning to Rank",
              "Recommendation Systems", "Embeddings", "Sentence Transformers"}
_NLP = {"NLP", "LLMs", "RAG", "Fine-tuning LLMs", "LoRA", "QLoRA", "PEFT",
        "Hugging Face Transformers", "LangChain", "LlamaIndex"}
_NOTE_WORDS = ["Watch", "Note", "Caveat", "Gap"]


def _recent_company(cand):
    for j in (cand.get("career_history") or []):
        if j.get("company"):
            return j["company"]
    return None


def _gap_years(cand):
    """Years since the most recent role ended (0 if currently employed)."""
    months = []
    for j in (cand.get("career_history") or []):
        if j.get("is_current"):
            return 0.0
        end = j.get("end_date")
        if end:
            try:
                d = datetime.strptime(end, "%Y-%m-%d")
                months.append((CURRENT_DATE.year - d.year) * 12 + (CURRENT_DATE.month - d.month))
            except ValueError:
                pass
    return round(min(months) / 12.0, 1) if months else 0.0


def extract_facts(cand, jd):
    """The literal values the reasoning is allowed to reference."""
    p = cand.get("profile") or {}
    canon = canonicalise_set(cand.get("skills"))
    s = cand.get("redrob_signals") or {}
    return {
        "title": p.get("current_title") or "Candidate",
        "years": p.get("years_of_experience"),
        "company": _recent_company(cand),
        "matched": sorted(canon & jd["required_skills"]),
        "missing": sorted(jd["required_skills"] - canon),
        "gap_years": _gap_years(cand),
        "response_rate": s.get("recruiter_response_rate"),
        "open_to_work": s.get("open_to_work_flag"),
    }


def _domain_word(matched):
    m = set(matched)
    if m & _RETRIEVAL:
        return "retrieval/ranking"
    if m & _NLP:
        return "NLP/LLM"
    return "applied-ML"


def _verdict(rank):
    if rank <= 10:
        return "Excellent fit."
    if rank <= 25:
        return "Strong fit."
    if rank <= 50:
        return "Solid fit."
    if rank <= 75:
        return "Plausible fit — worth a look."
    return "Boundary candidate — review before shortlisting."


def _strengths(f, dom, exp, prod, ten, avail, rec, sen, loc):
    """Up to 3 real strengths, each gated by the candidate's actual features."""
    out = []
    if dom >= 0.85 and f["matched"]:
        out.append(f"core {_domain_word(f['matched'])} work ({', '.join(f['matched'][:2])})")
    elif f["matched"]:
        out.append(f"hands-on with {', '.join(f['matched'][:2])}")
    if exp >= 0.95:
        out.append("squarely in the 6–8y band")
    if prod >= 0.85:
        out.append("product-company track record")
    elif ten >= 0.85:
        out.append("stable tenure")
    if avail >= 0.7 and f["open_to_work"]:
        out.append("responsive & open to work")
    elif rec >= 0.85:
        out.append("recently active")
    if len(out) < 3 and sen >= 1.0:
        out.append("seniority on target")
    if len(out) < 3 and loc >= 0.85:
        out.append("in a preferred location")
    return out[:3]


def _concern(f, dom, exp, prod, avail, rec, coh):
    """The single most salient honest concern (or None)."""
    rr = f["response_rate"]
    years = f["years"]
    if coh < 0.6:
        return "timeline/consistency flags to verify"
    if f["open_to_work"] is False:
        return "not currently open to work"
    if avail < 0.4:
        return (f"low recruiter engagement ({rr:.0%})"
                if isinstance(rr, (int, float)) else "low recruiter engagement")
    if rec < 0.6:
        return f"inactive ~{f['gap_years']}y"
    if years is not None and years > 8.5:
        return "slightly past the 6–8y sweet spot"
    if years is not None and years < 5:
        return "a little junior for the 6–8y band"
    if prod < 0.4:
        return "largely services/consulting background"
    if dom < 0.55:
        return "adjacent to the core retrieval/ranking focus"
    if f["missing"]:
        return f"gaps on {', '.join(f['missing'][:2])}"
    return None


def generate_reasoning(rank, cand, jd, feats):
    """Build a fact-grounded, rank-consistent, high-variety reasoning string."""
    f = extract_facts(cand, jd)

    def g(k, d=0.0):
        v = feats.get(k) if hasattr(feats, "get") else feats[k]
        try:
            return float(v)
        except (TypeError, ValueError):
            return d

    dom, exp, prod = g("domain_fit"), g("experience_fit"), g("product_ratio")
    ten, avail, rec = g("tenure_stability"), g("availability_score"), g("recency_score", 1.0)
    loc, coh, sen = g("location_fit"), g("coherence_score", 1.0), g("seniority_fit")

    strengths = _strengths(f, dom, exp, prod, ten, avail, rec, sen, loc)
    concern = _concern(f, dom, exp, prod, avail, rec, coh)

    at_co = f" at {f['company']}" if f["company"] else ""
    head = f["title"] + (f" ({f['years']}y)" if f["years"] is not None else "") + at_co
    body = "; ".join(strengths) if strengths else "relevant background"
    note = f" {_NOTE_WORDS[rank % len(_NOTE_WORDS)]}: {concern}." if concern else ""
    return f"{head} — {body}.{note} {_verdict(rank)}"
