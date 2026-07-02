"""
Stage 2 — Coherence Validator (hybrid: deterministic rules + soft anomaly signal)

Design (per project plan, reconciled with the IsolationForest work):

  * The HARD signal -> rank_ceiling comes from DETERMINISTIC impossibility checks.
    These need no ground-truth labels because they encode arithmetic / physical
    impossibility (claiming more experience than time since graduation, overlapping
    full-time jobs, future dates, joining a company before it existed, expert skills
    with no usage). A fabrication violates these as a side effect regardless of which
    honeypot "type" the organisers planted.

  * The SOFT signal is the IsolationForest anomaly score. It is NOT used to ban or
    ceiling anyone. It is carried forward as a feature column (`anomaly_score`) so the
    Stage 4 LightGBM ranker can weigh "statistical weirdness" as a gradient instead of
    a guillotine. This preserves the ability to catch unknown patterns without the
    22k-false-positive problem the pure-IForest ban list produced.

Output per candidate:
  coherence_score : float in [0, 1]   (1.0 = perfectly self-consistent)
  coherence_flags : list[str]          (which checks tripped)
  rank_ceiling    : int|None           (95 if score < CEILING_THRESHOLD else None)
  + the 5 meta-features and the soft anomaly_score

Writes:
  coherence_scores.csv   — one row per candidate, consumed by Stage 3/4
  coherence_ceiling.json — list of candidate_ids that received a rank_ceiling
"""

import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
# NOTE: scikit-learn is imported lazily inside add_anomaly_score() so that the
# deterministic validator (validate_coherence) can be imported WITHOUT sklearn —
# e.g. by the lightweight sandbox app, which doesn't need the anomaly signal.

# --------------------------------------------------------------------------- #
# Constants / tunable thresholds (calibrate against the score distribution)
# --------------------------------------------------------------------------- #
CURRENT_YEAR = 2026
CURRENT_DATE = datetime(CURRENT_YEAR, 6, 1)

CEILING_THRESHOLD = 0.40          # below this -> rank_ceiling = 95
RANK_CEILING_VALUE = 95

# Per-check penalty caps and tolerances. Tuned so that only ~50-150 candidates
# (vs ~80 planted honeypots) fall below CEILING_THRESHOLD. See calibrate notes.
FUTURE_DATE_PENALTY = 1.00        # any future date is a near-certain fabrication

# Career-timeline check: experience cannot exceed the span since the FIRST job.
CAREER_GRACE_YEARS = 1.5          # rounding + pre-first-listed-job work
CAREER_PER_YEAR = 0.20
CAREER_CAP = 0.70

# Pre-graduation check anchors on the EARLIEST degree (not the latest) so that a
# mid-career Master's/MBA never makes a legitimate senior look impossible.
POSTGRAD_GRACE_YEARS = 3.0        # may start working a few years before first degree
POSTGRAD_PER_YEAR = 0.15
POSTGRAD_CAP = 0.60

OVERLAP_GRACE_RATIO = 1.20        # up to 20% cumulative-vs-calendar overlap is fine
OVERLAP_PER_UNIT = 0.80           # penalty per 1.0 of ratio above grace
OVERLAP_CAP = 0.60

EXPERT_PER_SKILL = 0.12           # per expert-skill with < 12 months usage
EXPERT_CAP = 0.50

FOUNDING_PENALTY = 0.90           # started at a (real) company before it was founded

# Founding years for the REAL companies in the dataset. Fictional companies
# (Pied Piper, Initech, Wayne Enterprises, Stark Industries, Hooli, Globex,
# Acme Corp, Dunder Mifflin) are intentionally absent -> they are skipped, never
# producing a false positive.
COMPANY_FOUNDING_YEAR = {
    "Infosys": 1981, "Wipro": 1945, "TCS": 1968, "HCL": 1976,
    "Capgemini": 1967, "Accenture": 1989, "Cognizant": 1994,
    "Tech Mahindra": 1986, "Mphasis": 1998, "Mindtree": 1999, "Genpact": 1997,
    "Swiggy": 2014, "Zomato": 2008, "Flipkart": 2007, "Razorpay": 2014,
    "CRED": 2018, "Meesho": 2015, "Nykaa": 2012, "InMobi": 2007,
    "BYJU'S": 2011, "PolicyBazaar": 2008, "Ola": 2010, "Zoho": 1996,
    "Vedantu": 2011, "Paytm": 2010, "Unacademy": 2015, "PharmEasy": 2015,
    "upGrad": 2015, "Freshworks": 2010, "PhonePe": 2015, "Dream11": 2008,
    "Glance": 2019,
}

META_FEATURES = [
    "time_fabrication_ratio",
    "calendar_career_span_months",
    "post_grad_experience_ratio",
    "unearned_expert_count",
    "skill_inflation_index",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def parse_date(date_str):
    """Parse YYYY-MM-DD; fall back to CURRENT_DATE for missing/unparseable."""
    if not date_str:
        return CURRENT_DATE
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return CURRENT_DATE


def _year_of(date_str):
    """Return the integer year from a YYYY-... string, or None."""
    if not date_str or not isinstance(date_str, str) or len(date_str) < 4:
        return None
    try:
        return int(date_str[:4])
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# Feature extraction (shared with the inspector, single source of truth here)
# --------------------------------------------------------------------------- #
def extract_meta_features(data):
    """Compute the 5 coherence meta-features for one candidate dict."""
    profile = data.get("profile") or {}
    history = data.get("career_history") or []
    education = data.get("education") or []
    skills = data.get("skills") or []

    years_exp = float(profile.get("years_of_experience") or 0.0)

    total_career_months = 0
    start_dates, end_dates = [], []
    for job in history:
        total_career_months += int(job.get("duration_months") or 0)
        start_dates.append(parse_date(job.get("start_date")))
        end_dates.append(CURRENT_DATE if job.get("is_current")
                         else parse_date(job.get("end_date")))

    if start_dates and end_dates:
        absolute_start, absolute_end = min(start_dates), max(end_dates)
        calendar_span = max(
            1,
            (absolute_end.year - absolute_start.year) * 12
            + (absolute_end.month - absolute_start.month),
        )
        earliest_job_year = absolute_start.year
    else:
        calendar_span = 1
        earliest_job_year = None

    time_fabrication_ratio = max(1.0, total_career_months / calendar_span)

    # Earliest graduation anchors the pre-graduation check. Using the MINIMUM
    # positive end_year means a later degree never shrinks the allowable span.
    edu_end_years = [e.get("end_year") for e in education
                     if e.get("end_year") and int(e["end_year"]) > 0]
    if edu_end_years:
        earliest_grad_year = min(int(y) for y in edu_end_years)
        years_since_grad = max(1, CURRENT_YEAR - earliest_grad_year)
    else:
        years_since_grad = max(1, int(years_exp))  # fallback: no education listed

    post_grad_experience_ratio = max(1.0, years_exp / years_since_grad)

    unearned_expert_count = 0
    total_skill_months = 0
    for skill in skills:
        dur = int(skill.get("duration_months") or 0)
        total_skill_months += dur
        if skill.get("proficiency") == "expert" and dur < 12:
            unearned_expert_count += 1
    skill_inflation_index = total_skill_months / max(1, total_career_months)

    return {
        "time_fabrication_ratio": time_fabrication_ratio,
        "calendar_career_span_months": calendar_span,
        "post_grad_experience_ratio": post_grad_experience_ratio,
        "unearned_expert_count": unearned_expert_count,
        "skill_inflation_index": skill_inflation_index,
        # raw values reused by the deterministic checks below
        "_years_exp": years_exp,
        "_years_since_grad": years_since_grad,
        "_earliest_job_year": earliest_job_year,
    }


# --------------------------------------------------------------------------- #
# Deterministic impossibility checks -> penalties + flags
# --------------------------------------------------------------------------- #
def check_future_dates(data):
    """Any date after the current year is a near-certain fabrication."""
    for job in data.get("career_history") or []:
        for key in ("start_date", "end_date"):
            yr = _year_of(job.get(key))
            if yr and yr > CURRENT_YEAR:
                return FUTURE_DATE_PENALTY, "future_date"
    for edu in data.get("education") or []:
        for key in ("start_year", "end_year"):
            yr = edu.get(key)
            if yr and int(yr) > CURRENT_YEAR:
                return FUTURE_DATE_PENALTY, "future_date"
    for cert in data.get("certifications") or []:
        yr = cert.get("year")
        if yr and int(yr) > CURRENT_YEAR:
            return FUTURE_DATE_PENALTY, "future_date"
    return 0.0, None


def check_career_timeline(feats):
    """Claimed experience cannot exceed the span since the candidate's first job."""
    earliest = feats["_earliest_job_year"]
    if not earliest:
        return 0.0, None
    max_possible = CURRENT_YEAR - earliest
    delta = feats["_years_exp"] - max_possible
    if delta <= CAREER_GRACE_YEARS:
        return 0.0, None
    penalty = min(CAREER_CAP, (delta - CAREER_GRACE_YEARS) * CAREER_PER_YEAR)
    return penalty, "timeline_impossible"


def check_postgrad_timeline(feats):
    """Claiming materially more experience than years since the FIRST degree."""
    delta = feats["_years_exp"] - feats["_years_since_grad"]
    if delta <= POSTGRAD_GRACE_YEARS:
        return 0.0, None
    penalty = min(POSTGRAD_CAP, (delta - POSTGRAD_GRACE_YEARS) * POSTGRAD_PER_YEAR)
    return penalty, "experience_predates_education"


def check_job_overlap(feats):
    """Cumulative job-months far exceeding the calendar span => overlapping FT roles."""
    ratio = feats["time_fabrication_ratio"]
    if ratio <= OVERLAP_GRACE_RATIO:
        return 0.0, None
    penalty = min(OVERLAP_CAP, (ratio - OVERLAP_GRACE_RATIO) * OVERLAP_PER_UNIT)
    return penalty, "job_overlap"


def check_skill_inflation(feats):
    """Expert-level skills with under a year of usage."""
    count = feats["unearned_expert_count"]
    if count <= 0:
        return 0.0, None
    penalty = min(EXPERT_CAP, count * EXPERT_PER_SKILL)
    return penalty, "skill_inflation"


def check_company_founding(data):
    """Started at a known real company before it existed."""
    for job in data.get("career_history") or []:
        founded = COMPANY_FOUNDING_YEAR.get(job.get("company"))
        if not founded:
            continue
        start_yr = _year_of(job.get("start_date"))
        if start_yr and start_yr < founded:
            return FOUNDING_PENALTY, "company_predates_founding"
    return 0.0, None


def validate_coherence(data):
    """Run all checks for one candidate -> coherence dict."""
    feats = extract_meta_features(data)

    penalties = []
    flags = []
    for penalty, flag in (
        check_future_dates(data),
        check_career_timeline(feats),
        check_postgrad_timeline(feats),
        check_job_overlap(feats),
        check_skill_inflation(feats),
        check_company_founding(data),
    ):
        if penalty > 0:
            penalties.append(penalty)
            if flag:
                flags.append(flag)

    coherence_score = float(np.clip(1.0 - sum(penalties), 0.0, 1.0))
    rank_ceiling = RANK_CEILING_VALUE if coherence_score < CEILING_THRESHOLD else None

    return {
        "candidate_id": data.get("candidate_id"),
        "coherence_score": round(coherence_score, 4),
        "coherence_flags": flags,
        "rank_ceiling": rank_ceiling,
        "time_fabrication_ratio": round(feats["time_fabrication_ratio"], 4),
        "calendar_career_span_months": feats["calendar_career_span_months"],
        "post_grad_experience_ratio": round(feats["post_grad_experience_ratio"], 4),
        "unearned_expert_count": feats["unearned_expert_count"],
        "skill_inflation_index": round(feats["skill_inflation_index"], 4),
    }


# --------------------------------------------------------------------------- #
# Soft anomaly signal (IsolationForest) — feature only, never a ban
# --------------------------------------------------------------------------- #
def add_anomaly_score(df):
    """
    Add a continuous `anomaly_score` in [0, 1] (higher = more anomalous) using an
    ensemble of IsolationForests for stability. This is a SOFT feature for Stage 4,
    not a filter.
    """
    from sklearn.ensemble import IsolationForest  # lazy: keeps validate_coherence sklearn-free
    X = df[META_FEATURES].to_numpy(dtype=float)
    seeds = [42, 51, 88, 101, 212]
    scores = np.zeros(len(df), dtype=float)
    for seed in seeds:
        model = IsolationForest(
            n_estimators=100, contamination="auto", random_state=seed, n_jobs=-1
        )
        model.fit(X)
        # score_samples: higher = more normal. Negate so higher = more anomalous.
        scores += -model.score_samples(X)
    scores /= len(seeds)

    lo, hi = scores.min(), scores.max()
    df["anomaly_score"] = (scores - lo) / (hi - lo) if hi > lo else 0.0
    return df


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(input_path="candidates.jsonl", out_csv="coherence_scores.csv",
        out_ceiling="coherence_ceiling.json", sample=None):
    print(f"Stage 2 — Coherence Validator")
    print(f"Reading {input_path} ...")

    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not data.get("candidate_id"):
                continue
            rows.append(validate_coherence(data))
            if (i + 1) % 20000 == 0:
                print(f"  validated {i + 1} candidates...")
            if sample and len(rows) >= sample:
                break

    df = pd.DataFrame(rows)
    print(f"\nValidated {len(df)} candidates.")

    print("Computing soft anomaly score (IsolationForest ensemble)...")
    df = add_anomaly_score(df)

    # --- Distribution report (calibration aid) ---
    below = (df["coherence_score"] < CEILING_THRESHOLD).sum()
    print("\n================ COHERENCE DISTRIBUTION ================")
    for lo in [0.0, 0.2, 0.4, 0.6, 0.8]:
        hi = lo + 0.2
        n = ((df["coherence_score"] >= lo) & (df["coherence_score"] < hi)).sum()
        print(f"  score [{lo:.1f}, {hi:.1f}): {n:6d}")
    print(f"  score == 1.0    : {(df['coherence_score'] == 1.0).sum():6d}")
    print("-------------------------------------------------------")
    print(f"  RANK-CEILINGED (score < {CEILING_THRESHOLD}): {below}  "
          f"(target band ~50-150 vs ~80 planted)")
    print("=======================================================\n")

    # --- Flag frequency ---
    flag_counter = {}
    for flags in df["coherence_flags"]:
        for flg in flags:
            flag_counter[flg] = flag_counter.get(flg, 0) + 1
    print("Flag frequency (a candidate may trip several):")
    for flg, n in sorted(flag_counter.items(), key=lambda kv: -kv[1]):
        print(f"  {n:6d}  {flg}")

    # --- Persist ---
    df_out = df.copy()
    df_out["coherence_flags"] = df_out["coherence_flags"].apply(lambda x: "|".join(x))
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved per-candidate scores -> {out_csv}")

    ceiling_ids = df.loc[df["rank_ceiling"].notna(), "candidate_id"].tolist()
    with open(out_ceiling, "w") as f:
        json.dump(ceiling_ids, f)
    print(f"Saved {len(ceiling_ids)} rank-ceiling IDs -> {out_ceiling}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stage 2 coherence validator")
    ap.add_argument("--input", default="candidates.jsonl")
    ap.add_argument("--out", default="coherence_scores.csv")
    ap.add_argument("--ceiling-out", default="coherence_ceiling.json")
    ap.add_argument("--sample", type=int, default=None,
                    help="validate only the first N candidates (dev loop)")
    args = ap.parse_args()
    run(args.input, args.out, args.ceiling_out, args.sample)
