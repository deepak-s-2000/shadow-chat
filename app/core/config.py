from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "sqlite:///./chatbot.db"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
