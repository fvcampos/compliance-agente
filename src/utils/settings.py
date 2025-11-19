'''

Collection of functions and classes to manage application settings.

'''

import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    # 1. APP CONFIG
    APP_NAME: str = "ComplianceAgent"

    # "False" by default for safety (Production-first mindset)
    debug: bool = False

    # 2. INFRASTRUCTURE (Split Host/Port for flexibility)
    # "localhost" is the sensible default for Dev
    QDRANT_HOST: str = "localhost" 
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "compliance_docs"

    OPENAI_API_KEY: str | None = None

    @property
    def QDRANT_URL(self) -> str:
        """Computed property: Assembles the URL dynamically."""
        return f"http://{self.QDRANT_HOST}:{self.QDRANT_PORT}"

    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True)

settings: Settings = Settings()
