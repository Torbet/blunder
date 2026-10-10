from dotenv import find_dotenv
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class APISettings(BaseModel):
    host: str
    port: int

    @property
    def url(self) -> str:
        protocol = "https" if self.port == 443 else "http"
        return f"{protocol}://{self.host}:{self.port}"


class IngestionSettings(BaseModel):
    batch_size: int
    concurrency: int


class Settings(BaseSettings):
    api: APISettings
    ingestion: IngestionSettings

    model_config = SettingsConfigDict(env_nested_delimiter="__", env_file=find_dotenv(), extra="ignore")


settings = Settings()
