PYTHON ?= python

.PHONY: rank
rank:
	$(PYTHON) rank.py --candidates ./candidates.jsonl --out ./submission.csv

.PHONY: precompute
precompute:
	$(PYTHON) src/pipeline/precompute.py --input ./candidates.jsonl --artifacts ./artifacts

.PHONY: train
train:
	$(PYTHON) src/pipeline/train_ranker.py --input ./candidates.jsonl --labels ./labels.jsonl --coherence ./coherence_scores.csv --ceiling ./coherence_ceiling.json --artifacts ./artifacts
