from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    EMBEDDING_PROVIDER: str = "openai"
    OPENAI_API_KEY: str = ""

    LLM_PROVIDER: str = "openai"
    OPENROUTER_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"

    HTML_DIR: str = "./data/instructions"
    CHROMA_DIR: str = "./data/chroma"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
