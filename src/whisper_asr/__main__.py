"""Run Whisper ASR pipeline: Parquet evaluation or WAV batch transcription.

Usage:
  python -m whisper_asr parquet
  python -m whisper_asr wav
"""

from __future__ import annotations

import argparse
import logging
import sys

from whisper_asr.config import Config
from whisper_asr.transcribe import WhisperTranscriber
from whisper_asr.correct import SpellCorrector
from whisper_asr.evaluate import Evaluator
from whisper_asr.audio import load_from_bytes, load_from_file
from whisper_asr.text import normalize as normalize_text

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


def cmd_parquet(cfg: Config) -> None:
    import pandas as pd

    if not cfg.parquet_path:
        sys.exit("PARQUET_PATH not set. Pass --parquet or export PARQUET_PATH")

    transcriber = WhisperTranscriber(cfg)
    corrector = SpellCorrector(cfg) if cfg.llm_correction else None
    evaluator = Evaluator(cfg)

    df = pd.read_parquet(cfg.parquet_path).head(cfg.parquet_rows)
    logger.info("Processing %d rows from %s", len(df), cfg.parquet_path)

    results = []
    for idx, row in df.iterrows():
        try:
            audio_bytes = row["audio"]["bytes"]
            waveform = load_from_bytes(audio_bytes, target_sr=cfg.target_sr)

            whisper_text = transcriber.transcribe(waveform)

            llm_text = (
                corrector.correct(whisper_text)
                if corrector else whisper_text
            )

            reference = normalize_text(row.get("sentence", row.get("transcription", "")))

            duration_ms = row.get("duration_ms", 0)
            results.append({
                "row_id": idx,
                "sentence": reference,
                "whisper_prediction": whisper_text,
                "llm_prediction": llm_text,
                "duration_s": float(duration_ms) / 1000.0 if duration_ms else 0.0,
                "wer": evaluator.wer(reference, whisper_text),
                "cer": evaluator.cer(reference, whisper_text),
                "levenshtein_similarity": evaluator.levenshtein_similarity(reference, whisper_text),
                "semantic_similarity": evaluator.semantic_similarity(reference, whisper_text),
            })

            logger.info(
                "[%d/%d] WER=%.4f CER=%.4f",
                idx + 1, cfg.parquet_rows,
                results[-1]["wer"] or 0.0,
                results[-1]["cer"] or 0.0,
            )

        except Exception:
            logger.exception("Error processing row %d", idx)

    out = pd.DataFrame(results)
    out.to_csv(cfg.parquet_output, index=False, encoding="utf-8-sig")
    logger.info("Done → %s", cfg.parquet_output)


def cmd_wav(cfg: Config) -> None:
    import os as _os

    input_dir = cfg.wav_input_dir
    if not input_dir:
        sys.exit("WAV_INPUT_DIR not set. Pass --input-dir or export WAV_INPUT_DIR")

    _os.makedirs(cfg.wav_output_dir, exist_ok=True)

    transcriber = WhisperTranscriber(cfg)

    wav_files = sorted(f for f in _os.listdir(input_dir) if f.lower().endswith(".wav"))
    if not wav_files:
        logger.warning("No .wav files found in %s", input_dir)
        return

    logger.info("Transcribing %d files from %s", len(wav_files), input_dir)
    for filename in wav_files:
        path = _os.path.join(input_dir, filename)
        try:
            waveform = load_from_file(path, sr=cfg.target_sr)
            text = transcriber.transcribe(waveform)

            out_path = _os.path.join(
                cfg.wav_output_dir,
                f"{_os.path.splitext(filename)[0]}_transcript.txt",
            )
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)

            logger.info("%s → %s", filename, out_path)
        except Exception:
            logger.exception("Error processing %s", filename)


def main() -> None:
    parser = argparse.ArgumentParser(description="Whisper ASR pipeline")
    parser.add_argument(
        "mode", choices=["parquet", "wav"],
        help="parquet: evaluate against references — wav: batch transcribe .wav files",
    )
    parser.add_argument("--parquet", help="Path to .parquet file")
    parser.add_argument("--rows", type=int, help="Number of rows to process")
    parser.add_argument("--output", help="Output CSV path (parquet mode)")
    parser.add_argument("--input-dir", help="Input directory with .wav files")
    parser.add_argument("--output-dir", help="Output directory for transcripts (wav mode)")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    _setup_logging(args.verbose)

    cfg = Config.from_env()
    # CLI overrides
    if args.parquet:
        cfg = cfg.__class__(**{**cfg.__dict__, "parquet_path": args.parquet})
    if args.rows:
        cfg = cfg.__class__(**{**cfg.__dict__, "parquet_rows": args.rows})
    if args.output:
        cfg = cfg.__class__(**{**cfg.__dict__, "parquet_output": args.output})
    if args.input_dir:
        cfg = cfg.__class__(**{**cfg.__dict__, "wav_input_dir": args.input_dir})
    if args.output_dir:
        cfg = cfg.__class__(**{**cfg.__dict__, "wav_output_dir": args.output_dir})

    if args.mode == "parquet":
        cmd_parquet(cfg)
    else:
        cmd_wav(cfg)


if __name__ == "__main__":
    main()
