"""LLM-based spelling correction via Ollama."""

from __future__ import annotations

import logging

from langchain_ollama import ChatOllama

from whisper_asr.config import Config

logger = logging.getLogger(__name__)

_CORRECT_PROMPT = """
متن زیر خروجی سیستم تبدیل گفتار به متن است.

قوانین:

- فقط غلط های املایی را اصلاح کن.
- نیم فاصله ها را اصلاح کن.
- علائم نگارشی را حذف کن.
- هیچ کلمه ای اضافه نکن.
- هیچ کلمه ای حذف نکن.
- ترتیب کلمات را تغییر نده.
- جمله را بازنویسی نکن.
- فقط متن اصلاح شده را برگردان.
- هیچ توضیحی ننویس.

متن:

{text}
"""


class SpellCorrector:
    def __init__(self, cfg: Config) -> None:
        base_url = f"http://{cfg.ollama_host}:{cfg.ollama_port}"
        self._llm = ChatOllama(
            model=cfg.ollama_model,
            base_url=base_url,
            temperature=cfg.ollama_temperature,
        )

    def correct(self, text: str) -> str:
        if not text.strip():
            return text
        try:
            resp = self._llm.invoke(_CORRECT_PROMPT.format(text=text))
            return resp.content.strip()
        except Exception:
            logger.exception("LLM correction failed")
            return text
