# Architecture overview

This project is organized around four production stages:

1. Ingestion: read and normalize candidate records.
2. Coherence validation: flag impossible profiles and honeypots.
3. Feature engineering: build the JD-grounded ranking features.
4. Ranking: retrieve with FAISS, score with LightGBM, and generate submission output.

Main entry points:
- Inference: [rank.py](../../rank.py)
- Offline pipeline modules: [src/pipeline](../../src/pipeline)
- Sandbox demo: [sandbox/app.py](../../sandbox/app.py)
