"""
Stage 4 (offline) — label_llm.py  — THE BROAD NET

Auto-labels candidates with a free LLM (Groq Llama-3.3-70B or Gemini 2.0 Flash)
using the exact approved rubric. Batched, resumable, provider-agnostic.

Tier 1 of the label-scaling blueprint: slam the FAISS contenders + stratified
negatives through a free LLM to give LightGBM a broad, structurally-sound base.
Claude (in-session) then hand-labels the decisive top-200 boundary separately.

Honeypots are NOT sent here (deterministic 0 at train time — the LLM does no math).

Usage:
  GROQ_API_KEY=...  python label_llm.py --provider groq  --queue label_queue.jsonl
  GEMINI_API_KEY=... python label_llm.py --provider gemini --limit 500
"""

import os
import re
import abc
import sys
import json
import time
import argparse
import urllib.request
import urllib.error

STORE = "labels.jsonl"

# --- The approved system prompt (reasoning-first; explicit batch contract; no
#     honeypot math — handled deterministically in Python). ---
SYSTEM_PROMPT = """\
You are a senior technical recruiter at Redrob AI screening candidates for ONE
specific role: Senior AI Engineer (Founding Team). Assign each candidate a
relevance score from 0 to 5 reflecting how well they fit THIS role - the way an
experienced hiring manager would, by reasoning about what the profile MEANS, not
by counting keywords.

WHAT THE ROLE ACTUALLY NEEDS
- Core: production experience in embeddings-based retrieval, vector/hybrid search,
  and ranking / recommendation / search systems, plus rigorous ranking evaluation
  (NDCG / MRR / MAP). Strong Python.
- Ideal: 6-8 years total (5-9 acceptable), with ~4-5 years in applied ML/AI at
  PRODUCT companies, having shipped an end-to-end ranking/search/recommendation
  system to real users at scale.
- Nice to have: LLM fine-tuning (LoRA/QLoRA/PEFT), learning-to-rank, HR/recruiting
  tech, distributed systems, open-source contributions.

STRONG NEGATIVE SIGNALS (down-rank hard, even if the skills list looks perfect)
- Off-domain career: the person's actual TITLES and role DESCRIPTIONS are in HR,
  sales, marketing, finance, accounting, civil/mechanical engineering, design,
  content, operations, or generic business analysis - not AI/ML/IR. A profile
  stuffed with AI skills but whose career is e.g. "Marketing Manager" is a TRAP -> 0-1.
- Entire career at IT-services/consulting firms (TCS, Infosys, Wipro, Accenture,
  Cognizant, Capgemini, HCL, Tech Mahindra, Mphasis, Mindtree, Genpact) with no
  product-company experience.
- Primary expertise in computer vision, speech, or robotics WITHOUT NLP/IR/
  retrieval exposure.
- Title-chasing: job-hopping roughly every ~1.5 years optimizing for title bumps.
- Senior who stopped writing code (moved to pure "architecture"/management) -
  this role writes code.
- Pure research/academic background with no production deployment.
- Behavioral non-availability: inactive for months, very low recruiter response
  rate, not open to work - a perfect-on-paper candidate who cannot actually be
  hired is worth less.

SCORING SCALE
5 - Exceptional / near-ideal: applied ML or retrieval/ranking/search work clearly
    demonstrated at product companies, right seniority (6-8y), shipped relevant
    systems, available. May lack trendy keywords if the history proves the work.
4 - Strong: solid AI/ML or IR/search/data background at product-type companies,
    right level, only minor gaps.
3 - Relevant (the "relevant" threshold): right domain and level but a notable gap
    (adjacent domain, some services exposure, slightly off experience band, or
    weaker availability).
2 - Adjacent/partial: software or data engineer without clear retrieval/ranking/ML
    focus, OR right domain but wrong level/availability.
1 - Weak: mostly off-domain or services-only with thin relevance; keyword-only.
0 - Irrelevant: off-domain career with no meaningful relevance to this role.

JUDGMENT RULES
- Weigh career-history TITLES and DESCRIPTIONS above the self-reported skills list;
  skills here are unreliable.
- Reward demonstrated systems work over vocabulary. A candidate whose history shows
  building a recommendation system at a product company is a fit even without the
  words "RAG" or "Pinecone".
- Use the FULL 0-5 range; be decisive and consistent. Do not cluster everyone at 2-3.
- When uncertain between two scores, let domain fit and product-vs-services decide.

OUTPUT
You will receive a batch of exactly N candidates. You MUST return a JSON array of
exactly N objects, one per candidate, in the order received. Do not skip, merge,
or invent candidates. For each, write the reasoning FIRST, then the score:
[{"candidate_id":"CAND_XXXXXXX","reasoning":"<one concise sentence, grounded in the profile>","relevance":<int 0-5>}]
Return ONLY the JSON array. No text outside it."""


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
class Provider(abc.ABC):
    @abc.abstractmethod
    def complete(self, payloads):
        """Return raw model text for a batch of payload dicts."""


class GroqProvider(Provider):
    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, model="llama-3.3-70b-versatile"):
        self.model = model
        self.key = os.environ.get("GROQ_API_KEY")
        if not self.key:
            sys.exit("Set GROQ_API_KEY (get one free at console.groq.com/keys).")

    def complete(self, payloads):
        user = (f"Here are exactly {len(payloads)} candidates as a JSON array. "
                f"Return a JSON array of exactly {len(payloads)} label objects.\n"
                + json.dumps(payloads, ensure_ascii=False))
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": user}],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        req = urllib.request.Request(
            self.URL, data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.key}",
                     "Content-Type": "application/json",
                     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) redrob-labeler/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        return data["choices"][0]["message"]["content"]


class GeminiProvider(Provider):
    def __init__(self, model="gemini-2.0-flash"):
        self.model = model
        self.key = os.environ.get("GEMINI_API_KEY")
        if not self.key:
            sys.exit("Set GEMINI_API_KEY (aistudio.google.com/app/apikey).")
        self.url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{model}:generateContent?key={self.key}")

    def complete(self, payloads):
        user = (f"Here are exactly {len(payloads)} candidates as a JSON array. "
                f"Return a JSON array of exactly {len(payloads)} label objects.\n"
                + json.dumps(payloads, ensure_ascii=False))
        body = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
        }
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        return data["candidates"][0]["content"]["parts"][0]["text"]


# --------------------------------------------------------------------------- #
# Parsing / store
# --------------------------------------------------------------------------- #
def extract_objects(text):
    """Robustly pull label objects out of a model response."""
    objs = []
    for m in re.finditer(r"\{[^{}]*\}", text, re.S):
        try:
            o = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if "candidate_id" in o and "relevance" in o:
            objs.append(o)
    return objs


def load_store():
    out = {}
    try:
        with open(STORE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    out[r["candidate_id"]] = r
    except FileNotFoundError:
        pass
    return out


def append_store(store, objs, source):
    n = 0
    with open(STORE, "a", encoding="utf-8") as w:
        for o in objs:
            cid = o.get("candidate_id")
            rel = o.get("relevance")
            if cid in store or cid is None or rel is None:
                continue
            try:
                rel = int(rel)
            except (ValueError, TypeError):
                continue
            if not 0 <= rel <= 5:
                continue
            rec = {"candidate_id": cid, "relevance": rel,
                   "reasoning": o.get("reasoning", ""), "source": source}
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            store[cid] = rec
            n += 1
    return n


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(provider_name, queue="label_queue.jsonl", batch=12, limit=None, model=None):
    provider = (GroqProvider(model or "llama-3.3-70b-versatile")
                if provider_name == "groq"
                else GeminiProvider(model or "gemini-2.0-flash"))

    payloads = [json.loads(l) for l in open(queue, encoding="utf-8") if l.strip()]
    store = load_store()
    todo = [p for p in payloads if p["candidate_id"] not in store]
    if limit:
        todo = todo[:limit]
    print(f"{len(store)} already labeled; {len(todo)} to label via {provider_name}.")

    done = 0
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        for attempt in range(5):
            try:
                text = provider.complete(chunk)
                objs = extract_objects(text)
                got = {o["candidate_id"] for o in objs}
                want = {p["candidate_id"] for p in chunk}
                added = append_store(store, objs, provider_name)
                done += added
                missing = want - got
                print(f"  batch {i//batch+1}: +{added} (missing {len(missing)}) "
                      f"total={len(store)}")
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = 5 * (attempt + 1)
                    print(f"  429 rate-limited; sleeping {wait}s..."); time.sleep(wait)
                else:
                    print(f"  HTTP {e.code}: {e.read()[:200]}"); time.sleep(3)
            except Exception as e:
                print(f"  error: {e}"); time.sleep(3)
    print(f"\nDone. Added {done} labels. Store total = {len(store)}.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["groq", "gemini"], default="groq")
    ap.add_argument("--queue", default="label_queue.jsonl")
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    run(args.provider, args.queue, args.batch, args.limit, args.model)
