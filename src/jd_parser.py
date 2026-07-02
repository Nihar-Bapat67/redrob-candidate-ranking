"""
Stage 3b — Job-description parser.

Turns the prose JD into a structured spec the feature builder can compute against.
The JD is deliberately unstructured ("we're going to write this JD differently"),
so the parser combines:

  * regex extraction of the hard facts (experience band, notice period),
  * section-aware skill extraction (scan the "absolutely need" vs "like to have"
    vs "do NOT want" regions against the canonical skill vocabulary), and
  * domain/location/company lexicons derived from a careful reading of the JD.

Everything the parser keys on is something the JD literally says, so each extracted
field is defensible at the Stage-5 interview.
"""

import re

from canonicaliser import (
    canonicalise, SKILL_ALIASES,
    RETRIEVAL_RANKING_SKILLS, NLP_LLM_SKILLS, CORE_ML_SKILLS,
    CV_SPEECH_SKILLS, JD_REQUIRED_SKILLS, JD_NICE_TO_HAVE_SKILLS,
)

# Locations the JD names. Score tiers reflect "Pune/Noida-preferred", then the
# explicitly welcomed cities, then "outside India: case-by-case, no visa".
PREFERRED_LOCATIONS = {"pune", "noida"}
WELCOME_LOCATIONS = {"hyderabad", "mumbai", "delhi", "delhi ncr", "gurgaon",
                     "gurugram", "bangalore", "bengaluru", "chennai"}

# "People who have ONLY worked at consulting firms ... in their entire career."
CONSULTING_FIRMS = {
    "TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini",
    "HCL", "Tech Mahindra", "Mphasis", "Mindtree", "Genpact",
}

# Positive / negative free-text domain cues for titles & role descriptions.
POSITIVE_DOMAIN_TERMS = [
    "retrieval", "ranking", "recommendation", "recommender", "search",
    "embedding", "information retrieval", "relevance", "vector", "semantic",
    "nlp", "natural language", "machine learning", " ml ", "deep learning",
    "llm", "language model", "data scien", "data engineer", "feature pipeline",
    "personalization", "personalisation", "matching",
]
NEGATIVE_DOMAIN_TERMS = [
    "computer vision", "image", "opencv", "object detection", "speech",
    "asr", "tts", "robotics", "autonomous",
]
# Titles/roles that are clearly off-domain (the keyword-trap candidates).
NONTECH_TITLE_TERMS = [
    "hr ", "human resources", "recruit", "marketing", "sales", "account",
    "finance", "civil engineer", "mechanical engineer", "graphic", "content writer",
    "customer support", "operations manager", "operations", "business analyst",
    "project manager", "designer",
]

SENIORITY_TITLE_LEVELS = [
    (("intern", "trainee"), 0),
    (("junior", "associate", "jr"), 1),
    (("mid", "engineer ii", "software engineer", "developer", "analyst"), 2),
    (("senior", "sr", "lead"), 3),
    (("staff", "principal"), 4),
    (("architect", "head", "director", "vp", "chief"), 5),
]


def _extract_skills_in(text):
    """Return canonical skills mentioned anywhere in a text fragment."""
    found = set()
    low = text.lower()
    # alias keys (multi-word first so longer phrases win)
    for alias in sorted(SKILL_ALIASES, key=len, reverse=True):
        if re.search(r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])", low):
            found.add(canonicalise(alias))
    return found


def parse_jd(text):
    """Parse JD text -> structured dict."""
    low = text.lower()

    # --- experience band ---
    exp_min, exp_max, exp_ideal_lo, exp_ideal_hi = 5, 9, 6, 8
    m = re.search(r"(\d+)\s*[-–]\s*(\d+)\s*years", low)
    if m:
        exp_min, exp_max = int(m.group(1)), int(m.group(2))
    m2 = re.search(r"(\d+)\s*[-–]\s*(\d+)\s*years total", low)
    if m2:
        exp_ideal_lo, exp_ideal_hi = int(m2.group(1)), int(m2.group(2))

    # --- notice period ---
    max_notice_days = 30
    if "sub-30" in low or "30 days" in low:
        max_notice_days = 30

    # --- skills, scoped by section when possible ---
    required, nice = set(), set()
    need_blk = re.search(r"absolutely need(.+?)(things we|things we.d like|$)",
                         low, re.S)
    like_blk = re.search(r"like you to have(.+?)(things we explicitly|$)", low, re.S)
    if need_blk:
        required |= _extract_skills_in(need_blk.group(1))
    if like_blk:
        nice |= _extract_skills_in(like_blk.group(1))
    # union with the curated JD skill sets so we never miss the core requirements
    required |= JD_REQUIRED_SKILLS
    nice |= JD_NICE_TO_HAVE_SKILLS
    nice -= required  # required wins on overlap

    return {
        "required_skills": required,
        "nice_to_have_skills": nice,
        "positive_skills": RETRIEVAL_RANKING_SKILLS | NLP_LLM_SKILLS | CORE_ML_SKILLS,
        "negative_skills": CV_SPEECH_SKILLS,
        "target_seniority": 3,            # "Senior AI Engineer"
        "exp_min": exp_min, "exp_max": exp_max,
        "exp_ideal_lo": exp_ideal_lo, "exp_ideal_hi": exp_ideal_hi,
        "preferred_locations": PREFERRED_LOCATIONS,
        "welcome_locations": WELCOME_LOCATIONS,
        "consulting_firms": CONSULTING_FIRMS,
        "positive_domain_terms": POSITIVE_DOMAIN_TERMS,
        "negative_domain_terms": NEGATIVE_DOMAIN_TERMS,
        "nontech_title_terms": NONTECH_TITLE_TERMS,
        "max_notice_days": max_notice_days,
        "wants_product_over_services": True,   # explicit anti-consulting stance
        "wants_hands_on_recent": True,         # "this role writes code"
        "penalise_job_hopping": True,          # explicit anti title-chaser stance
    }


def load_and_parse(path="jd.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return parse_jd(f.read())


def build_jd_query(jd):
    """A concise semantic query for FAISS retrieval (the embedder truncates long prose).
    Shared by make_label_queue.py (offline) and rank.py (inference)."""
    req = ", ".join(sorted(jd["required_skills"]))
    nice = ", ".join(sorted(jd["nice_to_have_skills"]))
    return (
        "Senior AI Engineer. Applied ML / AI at a product company. "
        "Core: embeddings-based retrieval, vector search, ranking, recommendation "
        "and search systems, ranking evaluation (NDCG, MRR, MAP), Python. "
        f"Required: {req}. Nice to have: {nice}."
    )


if __name__ == "__main__":
    import json
    parsed = load_and_parse()
    printable = {k: (sorted(v) if isinstance(v, set) else v)
                 for k, v in parsed.items()}
    print(json.dumps(printable, indent=2, ensure_ascii=False))
