from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class PathConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PATH_")
    root: Path = PROJECT_ROOT
    data: Path = Field(default_factory=lambda: PROJECT_ROOT / "data")
    models: Path = Field(default_factory=lambda: PROJECT_ROOT / "models")
    logs: Path = Field(default_factory=lambda: PROJECT_ROOT / "logs")
    fma_metadata: Path = Field(
        default_factory=lambda: PROJECT_ROOT / "data" / "raw" / "fma_metadata"
    )
    fma_small: Path = Field(
        default_factory=lambda: PROJECT_ROOT / "data" / "raw" / "fma_small"
    )
    fma_large: Path = Field(
        default_factory=lambda: PROJECT_ROOT / "data" / "raw" / "fma_large"
    )
    spotify: Path = Field(
        default_factory=lambda: PROJECT_ROOT / "data" / "raw" / "spotify-12m-songs"
    )
    audio_spotify :Path = Field(default_factory=lambda:PROJECT_ROOT/"data"/"processed"/"audio")
    processed :Path = Field(default_factory=lambda:PROJECT_ROOT/"data"/"processed")
    

    def model_post_init(self, __context):
        self.models.mkdir(exist_ok=True, parents=True)
        self.logs.mkdir(exist_ok=True, parents=True)


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__")

    paths: PathConfig = Field(default_factory=lambda: PathConfig())
