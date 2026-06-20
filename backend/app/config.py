from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    EMBEDDING_PROVIDER: str = "openai"
    OPENAI_API_KEY: str = ""

    LLM_PROVIDER: str = "openai"
    OPENROUTER_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"

    HTML_DIR: str = "./data/instructions"
    CHROMA_DIR: str = "./data/chroma"

    # Retrieval. RELEVANCE_THRESHOLD is a cosine distance (lower = more similar);
    # a chunk is considered relevant when its distance is below it. Calibrated on
    # the golden set with tools/calibrate_threshold.py: 0.70 keeps factual recall
    # at 1.00 while refusing 100% of off-topic queries at the retrieval stage
    # (0.75 let half of them through). Re-run the tool after big content changes.
    RELEVANCE_THRESHOLD: float = 0.70
    DENSE_POOL: int = 20   # dense candidates fetched before fusion
    BM25_POOL: int = 20    # BM25 candidates fetched before fusion

    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "./logs"
    LOG_PROMPTS: bool = False  # if true, include a truncated message preview in /chat logs

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
