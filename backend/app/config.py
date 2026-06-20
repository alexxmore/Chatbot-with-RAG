from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    EMBEDDING_PROVIDER: str = "openai"
    OPENAI_API_KEY: str = ""

    LLM_PROVIDER: str = "openai"
    OPENROUTER_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"

    HTML_DIR: str = "./data/instructions"
    CHROMA_DIR: str = "./data/chroma"

    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "./logs"
    LOG_PROMPTS: bool = False  # if true, include a truncated message preview in /chat logs

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
