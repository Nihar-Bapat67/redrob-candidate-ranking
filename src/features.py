"""
Stage 3c — Feature matrix construction.

Builds one numeric feature row per candidate, encoding the dimensions a recruiter
reading THIS JD would actually weigh. The dataset is an explicit keyword trap
(skills are assigned ~uniformly at random, so an "HR Manager" can list FAISS+RAG),
so the discriminative features are built from TITLE + career DESCRIPTIONS + company
type + behavioural signals — not from raw skill overlap.

Each feature maps to a literal JD statement (defensible at the Stage-5 interview):

  skill_match_pct      coverage of JD "absolutely need" skills (kept, but low-trust)
  nice_to_have_pct     coverage of JD "like to have" skills
  domain_fit           AI / IR / ranking / ML signal in title+descriptions, minus
                       CV/speech and minus off-domain (HR/Sales/Marketing/...) titles
                       -> the answer to "all AI keywords but title 'Marketing Manager'"
  seniority_fit        closeness to "Senior AI Engineer"
  experience_fit       closeness to the 6-8y ideal (5-9 acceptable)
  product_ratio        share of career NOT at services/consulting firms
                       -> "only worked at consulting firms ... not a fit"
  tenure_stability     penalises 1.5y job-hopping -> "title-chasers ... not a fit"
  coding_recency       hands-on IC role recently -> "this role writes code"
  recency_score        last_active decay -> availability
  availability_score   response rate / open-to-work / interview completion / saved
                       -> "hasn't logged in 6 months, 5% response = not available"
  location_fit         Pune/Noida preferred; named cities welcome; outside India low
  notice_fit           sub-30-day notice preferred
  coherence_score      Stage 2 (honeypot/self-consistency)
  anomaly_score        Stage 2 soft anomaly signal
"""

import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from canonicaliser import (
    canonicalise_set, RETRIEVAL_RANKING_SKILLS, NLP_LLM_SKILLS, CORE_ML_SKILLS,
)
from jd_parser import load_and_parse, CONSULTING_FIRMS

CURRENT_DATE = datetime(2026, 6, 1)

# Known product companies in the dataset (fictional tech cos + real product
# startups). Anything in CONSULTING_FIRMS or IT-Services industry counts as
# services; the rest defaults to non-services.
SERVICES_INDUSTRIES = {"it services", "consulting"}

AI_DATA_TITLE_TERMS = ("ml engineer", "machine learning", "ai engineer", "ai/ml",
                       "a.i", "data scientist", "data engineer", "analytics engineer",
                       "research engineer", "nlp", "ai research", "applied scientist",
                       "applied ml", "search engineer", "recommendation", "relevance",
                       "ranking", "retrieval", "search & ranking")
SWE_TITLE_TERMS = ("software engineer", "backend", "back end", "full stack", "fullstack",
                   "developer", "cloud engineer", "devops", "frontend", "mobile",
                   "qa engineer", "data analyst", "sde", "sre", "platform engineer")
# Clearly off-domain OR not-what-the-JD-wants (pure research/management/consulting).
OFFDOMAIN_TITLE_TERMS = ("hr ", "human resource", "recruit", "sales", "marketing",
                         "account", "civil engineer", "mechanical engineer",
                         "graphic", "content writer", "customer support",
                         "operations", "business analyst", "project manager",
                         "designer", "finance", "research assistant", "research intern",
                         "consultant", "cto", "ceo", "founder")
MANAGERIAL_TERMS = ("manager", "director", "head of", "vp ", "chief", "architect")
IC_ENG_TERMS = ("engineer", "developer", "scientist", "programmer")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def _months_between(a, b):
    return (a.year - b.year) * 12 + (a.month - b.month)


def title_seniority(title):
    """Map a job title to a numeric seniority level 0-5."""
    t = (title or "").lower()
    if any(k in t for k in ("intern", "trainee")):
        return 0
    if any(k in t for k in ("principal", "staff")):
        return 4
    if any(k in t for k in ("architect", "head of", "director", "vp ", "chief")):
        return 5
    if any(k in t for k in ("senior", "sr ", "lead")):
        return 3
    if any(k in t for k in ("junior", "associate", "jr ")):
        return 1
    return 2  # default mid


def classify_job_services(job):
    """True if a career_history entry is at a services/consulting employer."""
    if job.get("company") in CONSULTING_FIRMS:
        return True
    if (job.get("industry") or "").strip().lower() in SERVICES_INDUSTRIES:
        return True
    return False


def _title_domain_score(title):
    t = (title or "").lower()
    if any(k in t for k in AI_DATA_TITLE_TERMS):
        return 1.0
    if any(k in t for k in OFFDOMAIN_TITLE_TERMS):
        return 0.05
    if any(k in t for k in SWE_TITLE_TERMS):
        return 0.55
    return 0.3


# --------------------------------------------------------------------------- #
# Feature computations
# --------------------------------------------------------------------------- #
def domain_fit(candidate, jd):
    """AI/IR/ranking/ML consistency across title + descriptions, net of CV/speech."""
    profile = candidate.get("profile") or {}
    history = candidate.get("career_history") or []

    title_score = _title_domain_score(profile.get("current_title"))

    blob = " ".join([
        profile.get("headline") or "", profile.get("summary") or "",
        " ".join((j.get("title") or "") + " " + (j.get("description") or "")
                 for j in history),
    ]).lower()

    pos = sum(1 for term in jd["positive_domain_terms"] if term in blob)
    neg = sum(1 for term in jd["negative_domain_terms"] if term in blob)
    # normalise: presence of several positive cues saturates to 1.0
    desc_score = np.clip(pos / 6.0, 0, 1) - np.clip(neg / 4.0, 0, 0.5)
    desc_score = float(np.clip(desc_score, 0, 1))

    return round(float(np.clip(0.6 * title_score + 0.4 * desc_score, 0, 1)), 4)


def experience_fit(years, jd):
    y = float(years or 0)
    lo, hi = jd["exp_ideal_lo"], jd["exp_ideal_hi"]
    amin, amax = jd["exp_min"], jd["exp_max"]
    if lo <= y <= hi:
        return 1.0
    if amin <= y < lo:
        return round(0.7 + 0.3 * (y - amin) / max(1e-6, lo - amin), 4)
    if hi < y <= amax:
        return round(1.0 - 0.3 * (y - hi) / max(1e-6, amax - hi), 4)
    if y < amin:
        return round(max(0.1, 0.7 - (amin - y) * 0.15), 4)
    return round(max(0.1, 0.7 - (y - amax) * 0.10), 4)  # over-experienced


def product_ratio(history):
    """Share of career months NOT at services/consulting firms."""
    total = sum(int(j.get("duration_months") or 0) for j in history)
    if total <= 0:
        return 0.5
    services = sum(int(j.get("duration_months") or 0)
                   for j in history if classify_job_services(j))
    return round(1.0 - services / total, 4)


def tenure_stability(history):
    """Penalise short average tenure (the title-chaser signal)."""
    durs = [int(j.get("duration_months") or 0) for j in history if j.get("duration_months")]
    if not durs:
        return 0.5
    avg_years = (sum(durs) / len(durs)) / 12.0
    # <=1.0y -> 0.1 ; >=2.5y -> 1.0 ; linear between
    return round(float(np.clip((avg_years - 1.0) / (2.5 - 1.0) * 0.9 + 0.1, 0.1, 1.0)), 4)


def coding_recency(candidate):
    """Recently hands-on IC engineering vs moved-to-management/architecture."""
    profile = candidate.get("profile") or {}
    title = (profile.get("current_title") or "").lower()
    if any(k in title for k in IC_ENG_TERMS) and not any(
            k in title for k in ("architect", "director", "head of", "vp ", "chief")):
        base = 1.0
    elif any(k in title for k in MANAGERIAL_TERMS):
        base = 0.4
    else:
        base = 0.6
    # blend with how recently they were active
    return round(base * (0.5 + 0.5 * recency_score(candidate)), 4)


def recency_score(candidate):
    signals = candidate.get("redrob_signals") or {}
    last = _parse_date(signals.get("last_active_date"))
    if not last:
        return 0.5
    months = max(0, _months_between(CURRENT_DATE, last))
    return round(float(np.clip(1.0 - months * 0.06, 0.0, 1.0)), 4)


def availability_score(candidate):
    s = candidate.get("redrob_signals") or {}
    parts = [
        (float(s.get("recruiter_response_rate") or 0), 0.30),
        (1.0 if s.get("open_to_work_flag") else 0.0, 0.20),
        (float(s.get("interview_completion_rate") or 0), 0.15),
        (float(s.get("profile_completeness_score") or 0) / 100.0, 0.10),
        (np.clip(float(s.get("saved_by_recruiters_30d") or 0) / 10.0, 0, 1), 0.10),
        (np.clip(float(s.get("search_appearance_30d") or 0) / 250.0, 0, 1), 0.10),
        (max(0.0, float(s.get("offer_acceptance_rate") or 0)), 0.05),
    ]
    return round(float(sum(v * w for v, w in parts)), 4)


def location_fit(candidate, jd):
    profile = candidate.get("profile") or {}
    signals = candidate.get("redrob_signals") or {}
    loc = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()
    relocate = bool(signals.get("willing_to_relocate"))

    if any(c in loc for c in jd["preferred_locations"]):
        return 1.0
    if any(c in loc for c in jd["welcome_locations"]):
        return 0.85
    if country in ("india", ""):
        return round(min(0.8, 0.55 + (0.15 if relocate else 0.0)), 4)
    return round(0.25 + (0.10 if relocate else 0.0), 4)  # outside India, no visa


def notice_fit(candidate, jd):
    s = candidate.get("redrob_signals") or {}
    notice = s.get("notice_period_days")
    if notice is None:
        return 0.6
    notice = int(notice)
    if notice <= jd["max_notice_days"]:
        return 1.0
    return round(float(np.clip(1.0 - (notice - jd["max_notice_days"]) / 150.0, 0.3, 1.0)), 4)


def skill_coverage(candidate, jd):
    canon = canonicalise_set(candidate.get("skills"))
    req = jd["required_skills"]
    nice = jd["nice_to_have_skills"]
    sm = len(canon & req) / len(req) if req else 0.0
    nh = len(canon & nice) / len(nice) if nice else 0.0
    return round(sm, 4), round(nh, 4)


# --------------------------------------------------------------------------- #
# Row / matrix builders
# --------------------------------------------------------------------------- #
def build_feature_row(candidate, jd, coherence=None):
    profile = candidate.get("profile") or {}
    history = candidate.get("career_history") or []
    sm, nh = skill_coverage(candidate, jd)
    seniority = title_seniority(profile.get("current_title"))

    row = {
        "candidate_id": candidate.get("candidate_id"),
        "skill_match_pct": sm,
        "nice_to_have_pct": nh,
        "domain_fit": domain_fit(candidate, jd),
        "seniority_fit": round(1.0 - min(abs(seniority - jd["target_seniority"]), 3) / 3.0, 4),
        "experience_fit": experience_fit(profile.get("years_of_experience"), jd),
        "product_ratio": product_ratio(history),
        "tenure_stability": tenure_stability(history),
        "coding_recency": coding_recency(candidate),
        "recency_score": recency_score(candidate),
        "availability_score": availability_score(candidate),
        "location_fit": location_fit(candidate, jd),
        "notice_fit": notice_fit(candidate, jd),
    }
    # carry Stage 2 outputs
    if coherence is not None:
        row["coherence_score"] = coherence.get("coherence_score", 1.0)
        row["anomaly_score"] = coherence.get("anomaly_score", 0.0)
        row["rank_ceiling"] = coherence.get("rank_ceiling")
    return row


FEATURE_COLUMNS = [
    "skill_match_pct", "nice_to_have_pct", "domain_fit", "seniority_fit",
    "experience_fit", "product_ratio", "tenure_stability", "coding_recency",
    "recency_score", "availability_score", "location_fit", "notice_fit",
    "coherence_score", "anomaly_score",
]


def _load_coherence(path):
    """Load Stage-2 outputs into {candidate_id: {...}}."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"  WARNING: {path} not found - coherence/anomaly features will default.")
        return {}
    cols = ["coherence_score", "anomaly_score", "rank_ceiling"]
    have = [c for c in cols if c in df.columns]
    return {r["candidate_id"]: {c: r[c] for c in have}
            for _, r in df[["candidate_id"] + have].iterrows()}


def build_feature_matrix(input_path="candidates.jsonl", jd_path="jd.txt",
                         coherence_path="coherence_scores.csv", sample=None):
    jd = load_and_parse(jd_path)
    coherence = _load_coherence(coherence_path)

    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                cand = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not cand.get("candidate_id"):
                continue
            coh = coherence.get(cand["candidate_id"])
            rows.append(build_feature_row(cand, jd, coh))
            if (i + 1) % 20000 == 0:
                print(f"  built features for {i + 1} candidates...")
            if sample and len(rows) >= sample:
                break

    df = pd.DataFrame(rows)
    # sensible defaults if coherence missing
    if "coherence_score" not in df:
        df["coherence_score"] = 1.0
    if "anomaly_score" not in df:
        df["anomaly_score"] = 0.0
    df["coherence_score"] = df["coherence_score"].fillna(1.0)
    df["anomaly_score"] = df["anomaly_score"].fillna(0.0)
    return df, jd


# --------------------------------------------------------------------------- #
# Driver + validation report
# --------------------------------------------------------------------------- #
def run(input_path="candidates.jsonl", jd_path="jd.txt",
        coherence_path="coherence_scores.csv", out="feature_matrix.csv", sample=None):
    print("Stage 3 — Feature Matrix Construction")
    df, jd = build_feature_matrix(input_path, jd_path, coherence_path, sample)
    print(f"\nBuilt {len(df)} feature rows x {len(FEATURE_COLUMNS)} features.")

    df.to_csv(out, index=False)
    print(f"Saved -> {out}")

    # ---- discrimination check: do the features separate real fits from the trap? ----
    print("\n=== Mean feature values by current_title (sanity check) ===")
    titles = {}
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if sample and i >= sample:
                break
            d = json.loads(line)
            titles[d["candidate_id"]] = (d.get("profile") or {}).get("current_title", "?")
    df["_title"] = df["candidate_id"].map(titles)
    watch = ["ML Engineer", "AI Research Engineer", "Data Scientist", "Data Engineer",
             "Backend Engineer", "Software Engineer", "HR Manager", "Marketing Manager",
             "Accountant", "Mechanical Engineer"]
    cols = ["domain_fit", "product_ratio", "seniority_fit", "experience_fit",
            "availability_score", "coherence_score"]
    summ = (df[df["_title"].isin(watch)].groupby("_title")[cols]
            .mean().reindex(watch).round(3))
    print(summ.to_string())
    print("\n(domain_fit should be high for ML/AI/Data titles, near-zero for "
          "HR/Marketing/Accountant/Mechanical — confirming the keyword trap is defused.)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stage 3 feature matrix builder")
    ap.add_argument("--input", default="candidates.jsonl")
    ap.add_argument("--jd", default="jd.txt")
    ap.add_argument("--coherence", default="coherence_scores.csv")
    ap.add_argument("--out", default="feature_matrix.csv")
    ap.add_argument("--sample", type=int, default=None)
    args = ap.parse_args()
    run(args.input, args.jd, args.coherence, args.out, args.sample)
