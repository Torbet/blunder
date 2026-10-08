from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class APISettings(BaseModel):
    port: int


class Settings(BaseSettings):
    api: APISettings

    model_config = SettingsConfigDict(env_nested_delimiter="__", env_file=".env", extra="ignore")


settings = Settings()
