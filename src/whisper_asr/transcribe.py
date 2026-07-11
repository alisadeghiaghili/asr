"""Whisper transcription pipeline."""

from __future__ import annotations

import logging

import numpy as np
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline as hf_pipeline,
)

from whisper_asr.config import Config

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    def __init__(self, cfg: Config) -> None:
        torch_dtype = torch.float16 if cfg.is_cuda else torch.float32
        model_path_or_id = cfg.whisper_model_path or cfg.whisper_model_id

        logger.info("Loading Whisper from %s", model_path_or_id)
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path_or_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self._model.to(cfg.device)

        processor = AutoProcessor.from_pretrained(model_path_or_id)

        self._pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=self._model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=cfg.device,
        )
        self._cfg = cfg

    def transcribe(self, waveform: np.ndarray) -> str:
        """Transcribe mono audio array. Returns cleaned text."""
        result = self._pipe(
            {"array": waveform, "sampling_rate": self._cfg.target_sr},
            generate_kwargs={
                "language": self._cfg.language,
                "task": "transcribe",
            },
        )
        return result["text"].strip()
