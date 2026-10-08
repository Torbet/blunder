from dotenv import find_dotenv
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class APISettings(BaseModel):
    port: int


class PostgresSettings(BaseModel):
    host: str
    port: int
    user: str
    password: str
    database: str

    @property
    def url(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class Settings(BaseSettings):
    api: APISettings
    postgres: PostgresSettings

    model_config = SettingsConfigDict(
        env_nested_delimiter="__", env_file=find_dotenv(), extra="ignore"
    )


settings = Settings()
