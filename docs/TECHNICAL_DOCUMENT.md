# AI Candidate Ranking System
## Technical Design & Architecture Document

**Project:** Intelligent Candidate Discovery & Ranking — Redrob AI Hackathon
**Prepared by:** Team (ML Engineering)
**Audience:** Judging panel — this document explains the system end to end, in plain language.

---

## 0. How to read this document

This is written the way a product company writes an internal design document: it starts
with the problem, explains every idea in plain English before using its technical name,
and ends with the questions a reviewer is likely to ask. If you read it top to bottom,
you will be able to explain the entire system confidently — what we built, why we built
it that way, and what we would do next.

---

## 1. The problem in one paragraph

We are given a **job description (JD)** and a pool of **100,000 candidate profiles**. We
must return the **top 100 candidates**, ranked best-to-worst, the way a world-class human
recruiter would — not the way a naive keyword search would. The catch: during judging,
our program must run **completely offline** (no internet, no calling ChatGPT/Claude), on
an **ordinary CPU laptop with 16 GB RAM**, and finish in **under 5 minutes**.

So we are being asked for two things that usually fight each other: **the judgment of a
smart human**, delivered at **the speed and cost of a simple program**.

---

## 2. Why this is hard (the three traps)

The organizers deliberately built traps into the dataset. Understanding them is the key
to understanding every design decision we made.

**Trap 1 — The keyword trap.** Every candidate has a `skills` list, but the skills are
assigned almost at random. We measured it: all 133 possible skills appear roughly the
same number of times (~12,000 each). So a "Marketing Manager" can have *FAISS, RAG,
PyTorch* listed in their skills. A system that ranks by "how many AI keywords match" gets
fooled into putting marketing managers at the top. **Keyword matching is a trap, by
design.**

**Trap 2 — Honeypots.** About 80 profiles are "honeypots": fake candidates with
*impossible* histories — e.g. 8 years of experience at a company founded 3 years ago, or
an "expert" in 10 skills with 0 months of usage. If more than 10% of our top 100 are
honeypots, we are **disqualified**. These are designed to look like perfect matches to a
keyword system.

**Trap 3 — Meaning vs words.** The JD literally says: *"A candidate who has all the AI
keywords but whose title is 'Marketing Manager' is not a fit. A candidate who never wrote
'RAG' but built a recommendation system at a product company IS a fit."* So the right
answer requires reading what a profile **means**, not what words it contains.

The whole competition is really a test of one thing: **can your system read a résumé the
way a person does, while running as cheaply as a spreadsheet?**

---

## 3. Our big idea, in one sentence

> We use a powerful Large Language Model (an AI like Claude) **once, offline, before the
> clock starts**, to *teach* a tiny, fast model. At competition time we run only the tiny
> model. We get AI-quality judgment at spreadsheet speed.

This technique is called **knowledge distillation** — a large "teacher" model transfers
its judgment into a small "student" model. The teacher is expensive and slow; the student
is cheap and fast. After teaching, we throw away the teacher and ship the student.

Think of it like a master chef writing down their recipe. Cooking with the chef standing
next to you is expensive. But once the recipe is written, any cook can reproduce the dish
quickly. The **LightGBM model** is our recipe — a ~200 KB file on disk.

---

## 4. The architecture at a glance

We split the system into **two phases**. This separation is the single most important
design decision, so it is worth stating clearly.

| | **Phase A — Pre-computation (offline)** | **Phase B — Inference (the judged run)** |
|---|---|---|
| When | Before the competition, on our own machine | During judging |
| Time budget | Unlimited (hours are fine) | **≤ 5 minutes** |
| Internet | Allowed | **Forbidden** |
| What happens | Build a search index of all 100k candidates; use an LLM to create training data; train the small ranking model | Load pre-built files from disk; rank; write the answer CSV |
| Analogy | The chef writes the recipe | The cook makes the dish |

Everything expensive — embedding 100,000 profiles, calling an LLM thousands of times,
training the model — happens in Phase A. Phase B just loads a few files and runs.

**The inference flow (Phase B), step by step:**

```
Job Description
      │
      ▼
1. Embed the JD into a vector  ──►  2. FAISS search: 100,000 → top 500   (RECALL: cast a wide net)
                                              │
                                              ▼
                          3. Coherence check: flag honeypots among the 500
                                              │
                                              ▼
                          4. Build 14 features for each of the 500
                                              │
                                              ▼
                          5. LightGBM scores all 500                      (PRECISION: fine ranking)
                                              │
                                              ▼
                          6. Push honeypots to the bottom
                                              │
                                              ▼
                          7. Sort, take top 100, write reasons → submission.csv
```

Total time: about **30 seconds**. The 5-minute budget is never in danger.

---

## 5. The two-model design: recall then precision

A recruiter does two mental steps: first they **skim** the whole pile and pull out the
plausibly-relevant résumés (fast, rough), then they **read carefully** to rank the
shortlist (slow, precise). We built exactly that.

- **Step 1 — Recall (FAISS):** From 100,000 candidates, quickly grab the **500** most
  semantically similar to the JD. This is a wide net — we would rather include a few
  weak ones than miss a strong one.
- **Step 2 — Precision (LightGBM):** Among those 500, do the careful, fine-grained
  ranking using 14 hand-designed features.

Why not run the careful model on all 100,000? Because building 14 features for 100,000
people takes minutes, and we only have five. FAISS lets us do the expensive careful step
on just 500 people. **Recall makes it complete; precision makes it smart; together they
are fast.**

---

## 6. The five stages, explained

### Stage 1 — Data Ingestion
**What:** Read the 100,000-line data file, one candidate at a time, and clean it up
(make sure every record has the fields we need; make sure a skills field that is
sometimes a list and sometimes a string always becomes a list).

**Why it matters:** If bad data slips through here, it causes a crash three stages later
that is very hard to trace. We fix data problems in exactly one place, at the front door.

**Key idea — streaming:** We read the file line-by-line ("streaming") instead of loading
all 200 MB into memory at once, so memory stays low.

### Stage 2 — Coherence Validation (catching honeypots)
**What:** Give every candidate a **coherence score** from 0 to 1 that measures whether
their story is *logically possible*. Impossible profiles (honeypots) get a low score and
are blocked from the top ranks.

**Why we used rules, not machine learning, here:** This is a subtle and important point
for the panel. Honeypots are caught by **arithmetic**, not by pattern-matching. "This
company was founded in 2021, but the candidate says they worked there since 2016" is a
*contradiction*, not a *statistical oddity*. You cannot learn from data that 2016 is
before 2021 — it is simply true. So we wrote **deterministic if-then checks**:
- Claimed experience greater than the time since their first job → impossible.
- A future date anywhere → fabricated.
- Joined a real company before it existed → impossible.
- "Expert" in a skill with under a year of use → inflated.

Using rules here is not a shortcut — it is the *epistemically correct* tool. We apply
machine learning where machine learning belongs (ranking), and rules where rules belong
(checking facts).

**The result:** on the real data this flags ~226 of 100,000 as impossible (0.23%), and
they are demoted so they cannot appear in the top ranks. Honeypot rate in our top 100:
**0%.**

*(We also keep a second, softer signal — an "anomaly score" from an Isolation Forest — as
one input the ranking model can use. It never bans anyone; the deterministic rules do the
blocking.)*

### Stage 3 — Feature Engineering (the heart of the system)
**What:** Turn each candidate into **14 numbers** that capture the things a recruiter
actually cares about. This is where we defeat the keyword trap.

The most important feature is **`domain_fit`**. Instead of counting skill keywords (which
are random noise), we read the candidate's **job titles and the descriptions of what they
actually did**, and measure how much of it is real AI/ML/retrieval/ranking work. We
proved this works: an *ML Engineer* scores ~0.90 on `domain_fit`, while an *HR Manager
with the same AI skills listed* scores ~0.09. **The trap is defused.**

The full 14 features, in plain English:

| Feature | Plain-English meaning |
|---|---|
| `domain_fit` | Is their real work in AI/ML/search/ranking? (from titles + descriptions, not skills) |
| `skill_match_pct` | Fraction of required skills present (kept, but trusted little — it's noisy) |
| `nice_to_have_pct` | Fraction of preferred skills present |
| `seniority_fit` | Does their level match "Senior"? |
| `experience_fit` | How close to the ideal 6–8 years? |
| `product_ratio` | Share of career at product companies vs. consulting/services |
| `tenure_stability` | Do they stay in jobs, or hop every ~1.5 years? |
| `coding_recency` | Are they still hands-on, or moved fully into management? |
| `recency_score` | How recently were they active on the platform? |
| `availability_score` | Do they respond to recruiters / are they open to work? |
| `location_fit` | Are they in or willing to move to the target cities? |
| `notice_fit` | How soon can they join? |
| `coherence_score` | The honesty/consistency score from Stage 2 |
| `anomaly_score` | The soft outlier signal from Stage 2 |

Every one of these maps to a real line in the JD. When a judge asks "why this feature?",
the answer is always "because the JD asks for it."

### Stage 4 — Retrieval + Ranking (the engine)
**Retrieval (FAISS):** We convert every candidate's profile text into a list of numbers
called an **embedding** (a "vector"). Similar meanings produce nearby vectors. We store
all 100,000 vectors in a **FAISS index** — a data structure built for finding the nearest
vectors extremely fast. At inference we embed the JD, ask FAISS for the 500 nearest
candidates, and get them in about 2 seconds.

**Ranking (LightGBM):** For those 500, we feed the 14 features into **LightGBM**, a
gradient-boosted decision-tree model. It outputs a relevance score for each candidate; we
sort by that score. LightGBM is ideal here because it is tiny (~200 KB), fast, runs on CPU,
and is **interpretable** — we can show exactly which features drove a decision.

### Stage 5 — Reasoning Generation
**What:** For each of the top 100, write one honest sentence explaining the rank — e.g.
*"7.2y semantic search + ranking at Zomato; open and available — strong fit."*

**Why we do NOT call an LLM here:** Two reasons. (1) Calling an AI at inference is
forbidden (no internet). (2) More importantly, an LLM can *hallucinate* — it might write
"expert in Kubernetes" for someone who never mentioned it. Instead we use **slot-filling**:
we extract real values from the profile (title, years, employer, matched skills) and drop
them into rank-appropriate sentence templates. This makes hallucination **structurally
impossible** — we can only state facts we actually pulled from the record. The tone also
matches the rank (confident at #1, cautious at #95), which is exactly what the judges
check for in the manual review.

---

## 7. The models and frameworks (a plain-English glossary of what we used)

- **Embeddings / vectors:** A way to turn text into a list of numbers so that similar
  meanings are close together. Lets a computer measure "how related are these two texts?"
- **sentence-transformers (all-MiniLM-L6-v2):** The specific model we use to create
  embeddings. We chose MiniLM (small, fast) over the larger MPNet because on our CPU
  MPNet would have taken ~8 hours to embed 100k profiles; MiniLM does it in ~45 minutes
  with negligible quality loss for the retrieval step.
- **FAISS (Facebook AI Similarity Search):** A library that stores millions of vectors and
  finds the nearest ones in milliseconds. This is our "recall" engine.
- **LightGBM:** A fast, small, gradient-boosted tree model. This is our "ranking brain,"
  trained on the LLM's judgments. It is the ~200 KB file we actually ship.
- **The LLM labelers (Claude + Groq/Llama-3.3):** Large language models we used **only
  offline** to read candidate profiles and rate their relevance 0–5. These ratings became
  the training data for LightGBM. They are never called during the judged run.
- **Isolation Forest:** A classic outlier-detection algorithm; we use its score as one
  soft input, not as a filter.

---

## 8. How we trained the ranking model

The ranker is only as good as the examples it learns from. Here is exactly how we made
those examples.

1. **Build the shortlist to label.** We used FAISS to find the strongest ~1,500
   contenders for the JD, plus a spread of clearly weaker candidates for contrast.
2. **Ask LLMs to grade them.** We fed each candidate's profile to an LLM with a carefully
   written instruction ("act as a senior recruiter for this specific role; score 0–5;
   reason first, then give the number; judge by career, not by keywords"). The LLM
   returned a relevance grade.
   - We used **Groq (Llama-3.3-70B)** for breadth (fast, free) and **Claude** for the
     highest-stakes top-200 contenders where the 4-vs-5 distinction matters most.
3. **Handle honeypots deterministically.** We do not ask the LLM to do date arithmetic
   (LLMs are unreliable at it). Honeypots are labeled 0 by our Stage-2 rules.
4. **Train LightGBM** on these labeled examples (currently ~660 quality labels). The model
   learns which *combinations* of the 14 features predict high relevance.

**One honest, important detail (this shows engineering maturity):** the LLM instruction
we wrote is the single most valuable asset in the project, because the entire ranking
model is a distillation of *that* instruction. We treat it as a first-class artifact and
version it.

---

## 9. A real engineering story: finding and fixing a bug

We include this because being able to talk about a bug you found and fixed is what
separates an engineer from a script-runner.

**The symptom.** We stress-tested the system with a random file of 100 candidates. It
ranked a **Marketing Manager #1** and put research interns in the top 10. Clearly wrong.

**The diagnosis.** We printed the model's **feature importances** — a report of which
features the model actually relies on. We found: `experience_fit` had importance 17.0
while `domain_fit` had 0.7. The model was almost **ignoring domain** and ranking mostly by
years of experience. So a 7-year Marketing Manager beat a 6-year ML Engineer.

**The root cause (the interesting part).** We had trained the ranker only on candidates
that FAISS had already pre-filtered — and those are *all* AI-relevant. Within that group,
"is this person in the AI domain?" barely varies, so the model never learned to use it.
It had been quietly relying on FAISS to remove off-domain people for it. This is a
classic **train/serve mismatch** — the training data did not look like the data the model
would face in the open.

**The fix.** (1) We added ~300 clearly off-domain people (marketing, HR, mechanical
engineers) labeled 0, so the model *had* to learn to reject them. (2) We broadened the
domain detector to recognize titles like "Search Engineer" and "Applied Scientist." (3)
We switched the training objective to a more stable one so the model uses all 14 features
instead of collapsing onto one.

**The result.** `domain_fit` became the model's #1 signal. The Marketing Manager dropped
from #1 to #41. And — crucially — the *real* submission got **better**, not just
different: the new top 10 are all 6–8-year core AI/ML engineers at Microsoft, Meta, Apple,
Google, and Flipkart.

The lesson we can state to judges: *"We don't trust a model that looks good; we open it
up, read its feature importances, and make sure it's right for the right reasons."*

---

## 10. Results and evidence

- **Valid submission:** passes the official validator (exactly 100 rows, unique ranks,
  scores non-increasing, correct format).
- **Honeypots in top 100:** **0**.
- **Inference time:** ~30 seconds (budget is 5 minutes).
- **Top-10 quality:** all candidates are 6–8 years, in core retrieval/ranking/AI roles, at
  product companies — exactly the JD's "ideal candidate."
- **Trap defused:** average keyword-match score is 0.02 (proving keywords are noise), while
  `domain_fit` cleanly separates real fits (~0.9) from trap profiles (~0.09).

---

## 11. How anyone can see it working (the demos)

We built two ways to run the system live:

1. **Streamlit sandbox** (`sandbox/app.py`) — upload a candidate file, get a ranked table
   and a downloadable CSV. This satisfies the competition's "hosted demo" requirement.
2. **Interactive web dashboard** (`vercel-dashboard/`), deployable to Vercel — a polished
   front end where you upload the JD as a **PDF** and the candidate file, press **Rank**,
   and watch an animated pipeline (removing honeypots → retrieval → LightGBM → reasoning)
   before the ranked results and download button appear. The front end is a static page;
   the ranking runs in a small Python **serverless function**.

Both run the exact same ranking code as the real pipeline (on an uploaded sample; the full
100k FAISS step is the offline part).

---

## 12. How we meet every competition constraint

| Constraint | How we satisfy it |
|---|---|
| ≤ 5 minutes | ~30 seconds; the heavy work is pre-computed offline |
| CPU only, no GPU | FAISS-CPU, CPU PyTorch, LightGBM — none need a GPU |
| ≤ 16 GB RAM | Streaming reads; we keep only the 500 retrieved records in memory |
| No internet during ranking | The model + index are files on disk; LLMs are used only offline |
| ≤ 10% honeypots in top 100 | Deterministic coherence checks → 0% |
| Exact CSV format | Validated against the official validator |
| Reproducible from a single command | `python rank.py --candidates ... --out ...` |

---

## 13. The technology stack (one-line summary)

**Python** · **sentence-transformers / MiniLM** (embeddings) · **FAISS** (vector search) ·
**LightGBM** (learning-to-rank) · **Claude + Groq/Llama** (offline label generation) ·
**scikit-learn** (Isolation Forest) · **Streamlit** and **Vercel + serverless Python**
(demos) · **Git** (versioned, day-wise repository).

---

## 14. Questions a judge might ask — and confident answers

**Q: You used Claude/an LLM — isn't that against the "no API calls" rule?**
A: The rule forbids API calls *during ranking*. We use the LLM only in pre-computation, to
create training data — the same way any production ML team uses expensive tools at
training time and a cheap model at serving time. At judging, we call nothing; we load a
~200 KB file.

**Q: Why rules for honeypots instead of a classifier?**
A: Because a honeypot is a *logical impossibility*, not a *statistical pattern*. You can't
learn from data that 2016 is before 2021 — it's arithmetic. Rules are the correct tool for
checking facts; we save machine learning for ranking, where it belongs.

**Q: Why two models (FAISS + LightGBM) instead of one?**
A: They do different jobs. FAISS gives us **recall** — never miss a good candidate — over
100k people, fast. LightGBM gives us **precision** — fine-grained ranking — on the 500 that
matter. One without the other is either slow or shallow.

**Q: The skills list has all the AI keywords — how do you avoid being fooled?**
A: We measured that skills are assigned at random, so we barely use them. Our main signal,
`domain_fit`, reads job titles and role descriptions — what the person actually *did*. An
HR Manager with AI skills scores 0.09; an ML Engineer scores 0.90.

**Q: What would you do with more time?**
A: Three things: (1) generate more LLM labels to sharpen the ranking; (2) make `domain_fit`
distinguish *within* AI (e.g. retrieval vs. computer-vision) more finely; (3) add a second
FAISS query with a reworded JD to further improve recall.

**Q: How do we know the model is correct and not lucky?**
A: We read its feature importances and confirmed it relies on the right signals
(`domain_fit`, experience, availability). We also stress-tested it on adversarial data and
fixed a train/serve bug it exposed. We trust it because we opened it up, not because the
numbers looked nice.

---

*End of document.*
