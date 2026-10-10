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


class Settings(BaseSettings):
    api: APISettings

    model_config = SettingsConfigDict(
        env_nested_delimiter="__", env_file=find_dotenv(), extra="ignore"
    )


settings = Settings()
