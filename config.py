# config.py
import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field

class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""
    api_key: str = Field(..., description="OpenAI API key")
    gpt4_model: str = Field("gpt-4", description="GPT-4 model name")
    vision_model: str = Field("gpt-4-vision-preview", description="GPT-4 Vision model name")
    whisper_model: str = Field("whisper-1", description="Whisper model for transcription")
    max_tokens: int = Field(4096, description="Maximum tokens for completion")
    temperature: float = Field(0.0, description="Temperature for completions")
    timeout: int = Field(300, description="API timeout in seconds")

class ProcessingConfig(BaseModel):
    """Processing configuration."""
    chunk_size: int = Field(2000, description="Text chunk size")
    chunk_overlap: int = Field(200, description="Chunk overlap size")
    max_retries: int = Field(3, description="Maximum retry attempts")
    concurrent_files: int = Field(5, description="Maximum concurrent file processing")
    min_image_size: int = Field(50, description="Minimum image dimension in pixels")
    schema_path: str = Field("./schema/config_schema.json", description="Path to config schema file")

class Config(BaseModel):
    """Main configuration."""
    openai: OpenAIConfig
    processing: ProcessingConfig
    temp_dir: str = Field(default="./temp", description="Temporary file directory")
    output_dir: str = Field(default="./output", description="Output directory")
    log_dir: str = Field(default="./logs", description="Log directory")

def load_config() -> Config:
    """Load configuration from environment variables and .env file."""
    load_dotenv()

    # OpenAI configuration
    openai_config = OpenAIConfig(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        gpt4_model=os.getenv("OPENAI_GPT4_MODEL", "gpt-4"),
        vision_model=os.getenv("OPENAI_VISION_MODEL", "gpt-4-vision-preview"),
        whisper_model=os.getenv("OPENAI_WHISPER_MODEL", "whisper-1"),
        max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "4096")),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
        timeout=int(os.getenv("OPENAI_TIMEOUT", "300"))
    )

    # Processing configuration
    processing_config = ProcessingConfig(
        chunk_size=int(os.getenv("PROCESSING_CHUNK_SIZE", "2000")),
        chunk_overlap=int(os.getenv("PROCESSING_CHUNK_OVERLAP", "200")),
        max_retries=int(os.getenv("PROCESSING_MAX_RETRIES", "3")),
        concurrent_files=int(os.getenv("PROCESSING_CONCURRENT_FILES", "5")),
        min_image_size=int(os.getenv("PROCESSING_MIN_IMAGE_SIZE", "50")),
        schema_path=os.getenv("CONFIG_SCHEMA_PATH", "./schema/config_schema.json")
    )

    # Main configuration
    config = Config(
        openai=openai_config,
        processing=processing_config,
        temp_dir=os.getenv("TEMP_DIR", "./temp"),
        output_dir=os.getenv("OUTPUT_DIR", "./output"),
        log_dir=os.getenv("LOG_DIR", "./logs")
    )

    return config