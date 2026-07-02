"""
Stage 3a — Skill canonicaliser + domain lexicons.

Built from the ACTUAL 133-skill vocabulary in candidates.jsonl plus the synonyms
the JD uses in prose. Two jobs:

  1. canonicalise() collapses aliases ("k8s" -> "Kubernetes", "sentence-transformers"
     -> "Sentence Transformers") so skill comparisons are vocabulary-robust.

  2. Domain lexicons classify each skill into a domain bucket. This matters because
     the dataset deliberately assigns skills almost uniformly at random (every skill
     occurs ~12k times across 100k candidates), so a raw skill-overlap count is a
     trap — an "HR Manager" can list FAISS and RAG. The real signal is whether a
     candidate's TITLE and career DESCRIPTIONS are consistent with the JD's domain
     (NLP / information retrieval / ranking / ML), which the lexicons below feed.
"""

# --------------------------------------------------------------------------- #
# Alias -> canonical skill name. Keys are lower-cased; values are canonical.
# --------------------------------------------------------------------------- #
SKILL_ALIASES = {
    # vector search / databases
    "faiss": "FAISS", "pinecone": "Pinecone", "weaviate": "Weaviate",
    "qdrant": "Qdrant", "milvus": "Milvus", "opensearch": "OpenSearch",
    "elasticsearch": "Elasticsearch", "elastic search": "Elasticsearch",
    "es": "Elasticsearch", "pgvector": "pgvector", "vector search": "Vector Search",
    "vector database": "Vector Search", "vector db": "Vector Search",
    "semantic search": "Semantic Search",
    # embeddings / encoders
    "embeddings": "Embeddings", "embedding": "Embeddings",
    "sentence transformers": "Sentence Transformers",
    "sentence-transformers": "Sentence Transformers",
    "sbert": "Sentence Transformers", "text encoders": "Text Encoders",
    "vector representations": "Vector Representations",
    "bge": "Embeddings", "e5": "Embeddings",
    # retrieval / ranking / IR
    "information retrieval": "Information Retrieval",
    "information retrieval systems": "Information Retrieval",
    "ir": "Information Retrieval", "bm25": "BM25",
    "ranking systems": "Ranking Systems", "ranking": "Ranking Systems",
    "learning to rank": "Learning to Rank", "ltr": "Learning to Rank",
    "recommendation systems": "Recommendation Systems",
    "recommender systems": "Recommendation Systems",
    "recsys": "Recommendation Systems",
    "search & discovery": "Search & Discovery",
    "search backend": "Search Backend", "search infrastructure": "Search Infrastructure",
    "indexing algorithms": "Indexing Algorithms",
    # LLM / NLP
    "nlp": "NLP", "natural language processing": "NLP",
    "llm": "LLMs", "llms": "LLMs", "large language models": "LLMs",
    "rag": "RAG", "retrieval augmented generation": "RAG",
    "retrieval-augmented generation": "RAG",
    "fine-tuning llms": "Fine-tuning LLMs", "fine tuning": "Fine-tuning LLMs",
    "finetuning": "Fine-tuning LLMs", "lora": "LoRA", "qlora": "QLoRA",
    "peft": "PEFT", "prompt engineering": "Prompt Engineering",
    "model adaptation": "Model Adaptation",
    "hugging face transformers": "Hugging Face Transformers",
    "huggingface": "Hugging Face Transformers", "transformers": "Hugging Face Transformers",
    "langchain": "LangChain", "llamaindex": "LlamaIndex",
    "llama index": "LlamaIndex", "haystack": "Haystack",
    # core ML
    "machine learning": "Machine Learning", "ml": "Machine Learning",
    "deep learning": "Deep Learning", "dl": "Deep Learning",
    "data science": "Data Science", "scikit-learn": "scikit-learn",
    "sklearn": "scikit-learn", "pytorch": "PyTorch", "torch": "PyTorch",
    "tensorflow": "TensorFlow", "tf": "TensorFlow",
    "feature engineering": "Feature Engineering",
    "statistical modeling": "Statistical Modeling",
    "time series": "Time Series", "forecasting": "Forecasting",
    "reinforcement learning": "Reinforcement Learning", "rl": "Reinforcement Learning",
    # MLOps / infra
    "mlops": "MLOps", "mlflow": "MLflow", "kubeflow": "Kubeflow",
    "bentoml": "BentoML", "weights & biases": "Weights & Biases",
    "wandb": "Weights & Biases", "workflow orchestration": "Workflow Orchestration",
    # data eng
    "spark": "Spark", "pyspark": "Spark", "apache spark": "Spark",
    "apache beam": "Apache Beam", "beam": "Apache Beam",
    "apache flink": "Apache Flink", "flink": "Apache Flink",
    "kafka": "Kafka", "airflow": "Airflow", "dbt": "dbt", "etl": "ETL",
    "data pipelines": "Data Pipelines", "hadoop": "Hadoop",
    "snowflake": "Snowflake", "databricks": "Databricks", "bigquery": "BigQuery",
    # platforms / langs
    "python": "Python", "py": "Python", "java": "Java", "golang": "Go", "go": "Go",
    "rust": "Rust", "typescript": "TypeScript", "ts": "TypeScript",
    "javascript": "JavaScript", "js": "JavaScript",
    "aws": "AWS", "gcp": "GCP", "google cloud": "GCP", "azure": "Azure",
    "kubernetes": "Kubernetes", "k8s": "Kubernetes", "docker": "Docker",
    "ci/cd": "CI/CD", "microservices": "Microservices",
    "rest apis": "REST APIs", "rest": "REST APIs", "grpc": "gRPC",
    "graphql": "GraphQL", "postgresql": "PostgreSQL", "postgres": "PostgreSQL",
    "mongodb": "MongoDB", "redis": "Redis", "sql": "SQL",
    # CV / speech (negative-domain skills for THIS jd)
    "computer vision": "Computer Vision", "cv": "Computer Vision",
    "opencv": "OpenCV", "cnn": "CNN", "object detection": "Object Detection",
    "yolo": "YOLO", "image classification": "Image Classification",
    "diffusion models": "Diffusion Models", "gans": "GANs",
    "speech recognition": "Speech Recognition", "asr": "ASR", "tts": "TTS",
}

# --------------------------------------------------------------------------- #
# Domain buckets (canonical skill -> domain). Used by features.domain_fit.
# --------------------------------------------------------------------------- #
RETRIEVAL_RANKING_SKILLS = {
    "FAISS", "Pinecone", "Weaviate", "Qdrant", "Milvus", "OpenSearch",
    "Elasticsearch", "pgvector", "Vector Search", "Semantic Search",
    "Information Retrieval", "BM25", "Ranking Systems", "Learning to Rank",
    "Recommendation Systems", "Search & Discovery", "Search Backend",
    "Search Infrastructure", "Indexing Algorithms", "Embeddings",
    "Sentence Transformers", "Text Encoders", "Vector Representations",
}
NLP_LLM_SKILLS = {
    "NLP", "LLMs", "RAG", "Fine-tuning LLMs", "LoRA", "QLoRA", "PEFT",
    "Prompt Engineering", "Model Adaptation", "Hugging Face Transformers",
    "LangChain", "LlamaIndex", "Haystack",
}
CORE_ML_SKILLS = {
    "Machine Learning", "Deep Learning", "Data Science", "scikit-learn",
    "PyTorch", "TensorFlow", "Feature Engineering", "Statistical Modeling",
    "MLOps", "MLflow", "Kubeflow", "BentoML", "Weights & Biases",
}
# Negative domains for THIS jd: "primary expertise in CV, speech, or robotics
# without significant NLP/IR exposure" is an explicit anti-signal.
CV_SPEECH_SKILLS = {
    "Computer Vision", "OpenCV", "CNN", "Object Detection", "YOLO",
    "Image Classification", "Diffusion Models", "GANs",
    "Speech Recognition", "ASR", "TTS",
}
# Clearly non-engineering skills (the "Marketing Manager with AI keywords" trap).
NON_TECH_SKILLS = {
    "Accounting", "Sales", "Marketing", "SEO", "Content Writing", "Excel",
    "Tally", "SAP", "Salesforce CRM", "Six Sigma", "Project Management",
    "PowerPoint", "Figma", "Photoshop", "Illustrator",
}

# Skills the JD calls out as core "absolutely need" / "like to have".
JD_REQUIRED_SKILLS = (
    RETRIEVAL_RANKING_SKILLS | {"Python"} | {"Machine Learning", "Deep Learning"}
)
JD_NICE_TO_HAVE_SKILLS = NLP_LLM_SKILLS | {"Learning to Rank", "MLOps"}


def canonicalise(skill):
    """Map a raw skill string to its canonical form (best-effort)."""
    if not skill:
        return ""
    key = str(skill).strip().lower()
    if key in SKILL_ALIASES:
        return SKILL_ALIASES[key]
    # already canonical (title-cased dataset value) -> return as-is
    return str(skill).strip()


def canonicalise_set(skills):
    """Canonicalise an iterable of raw skill names/dicts into a set."""
    out = set()
    for s in skills or []:
        name = s.get("name") if isinstance(s, dict) else s
        canon = canonicalise(name)
        if canon:
            out.add(canon)
    return out
