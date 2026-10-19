from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = BACKEND_DIR / "models" / "battery"


class Settings(BaseSettings):
    app_name: str = "DriveGuard-X AI Backend"
    environment: str = "development"

    battery_model_path: Path = (
        DEFAULT_MODEL_DIR / "driveguard_x_final_battery_rf.joblib"
    )
    battery_metadata_path: Path = (
        DEFAULT_MODEL_DIR / "final_model_metadata.json"
    )
    battery_schema_path: Path = (
        DEFAULT_MODEL_DIR / "feature_schema.json"
    )

    cors_origins: list[str] = ["*"]
    mongodb_url: str = "mongodb://127.0.0.1:27017"
    mongodb_database: str = "driveguard_x"
    groq_api_key: str | None = None
    groq_model: str = "llama-3.1-8b-instant"
    recording_max_snapshots: int = 20000
    report_timeout_seconds: float = 30.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
