"""Configuration via environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    whisper_model_path: str = ""
    whisper_model_id: str = "openai/whisper-large-v3"
    device: str = "cuda:0"
    torch_dtype: str = "float16"
    target_sr: int = 16_000
    language: str = "persian"

    # ollama
    ollama_host: str = "localhost"
    ollama_port: str = "11434"
    ollama_model: str = "gemma-4-E4b-it:latest"
    ollama_temperature: float = 0.0
    # empty = no LLM correction
    llm_correction: bool = True

    # semantic model for evaluation
    semantic_model_path: str = ""

    # parquet defaults
    parquet_path: str = ""
    parquet_rows: int = 100
    parquet_output: str = "output.csv"

    # wav defaults
    wav_input_dir: str = ""
    wav_output_dir: str = "transcripts"

    @classmethod
    def from_env(cls) -> Config:
        return cls(
            whisper_model_path=os.environ.get("WHISPER_MODEL_PATH", ""),
            whisper_model_id=os.environ.get("WHISPER_MODEL_ID", "openai/whisper-large-v3"),
            device=os.environ.get("ASR_DEVICE", "cuda:0"),
            torch_dtype=os.environ.get("ASR_DTYPE", "float16"),
            target_sr=int(os.environ.get("ASR_TARGET_SR", "16000")),
            language=os.environ.get("ASR_LANGUAGE", "persian"),
            ollama_host=os.environ.get("OLLAMA_HOST", "localhost"),
            ollama_port=os.environ.get("OLLAMA_PORT", "11434"),
            ollama_model=os.environ.get("OLLAMA_MODEL", "gemma-4-E4b-it:latest"),
            ollama_temperature=float(os.environ.get("OLLAMA_TEMPERATURE", "0.0")),
            llm_correction=os.environ.get("LLM_CORRECTION", "true").lower() == "true",
            semantic_model_path=os.environ.get("SEMANTIC_MODEL_PATH", ""),
            parquet_path=os.environ.get("PARQUET_PATH", ""),
            parquet_rows=int(os.environ.get("PARQUET_ROWS", "100")),
            parquet_output=os.environ.get("PARQUET_OUTPUT", "output.csv"),
            wav_input_dir=os.environ.get("WAV_INPUT_DIR", ""),
            wav_output_dir=os.environ.get("WAV_OUTPUT_DIR", "transcripts"),
        )

    @property
    def is_cuda(self) -> bool:
        return self.device.startswith("cuda")
