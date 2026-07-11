"""Evaluation metrics: WER, CER, Levenshtein, semantic similarity."""

from __future__ import annotations

import logging

import numpy as np
from jiwer import cer, wer
from rapidfuzz.distance import Levenshtein
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from whisper_asr.config import Config

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, cfg: Config) -> None:
        path = cfg.semantic_model_path
        if path:
            logger.info("Loading semantic model from %s", path)
            self._semantic = SentenceTransformer(path)
        else:
            logger.info("Loading semantic model: paraphrase-multilingual-MiniLM-L12-v2")
            self._semantic = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )

    @staticmethod
    def wer(ref: str, hyp: str) -> float | None:
        try:
            return wer(ref, hyp)
        except Exception:
            return None

    @staticmethod
    def cer(ref: str, hyp: str) -> float | None:
        try:
            return cer(ref, hyp)
        except Exception:
            return None

    @staticmethod
    def levenshtein_similarity(ref: str, hyp: str) -> float:
        d = Levenshtein.distance(ref, hyp)
        max_len = max(len(ref), len(hyp))
        if max_len == 0:
            return 1.0
        return 1.0 - (d / max_len)

    def semantic_similarity(self, ref: str, hyp: str) -> float:
        emb = self._semantic.encode([ref, hyp])
        score = cosine_similarity(emb[0:1], emb[1:2])[0][0]
        return float(score)
