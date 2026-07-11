# whisper-asr

Persian ASR pipeline using **Whisper Large-v3** with optional **LLM spelling correction** via Ollama.

Two modes:

| Mode | Input | Output |
|------|-------|--------|
| `parquet` | Parquet file (audio bytes + reference text) | CSV with metrics (WER, CER, similarity) |
| `wav` | Directory of `.wav` files | One `.txt` per file |

## Quickstart

```bash
pip install -e .
```

### 1. Parquet (evaluation)

```bash
export WHISPER_MODEL_PATH=/path/to/whisper-large-v3/snapshots/xxx
export PARQUET_PATH=/path/to/data/train-00000-of-XXXXX.parquet

python -m whisper_asr parquet
```

Options: `--parquet --rows 50 --output results.csv --verbose`

### 2. WAV (batch transcribe)

```bash
export WHISPER_MODEL_PATH=/path/to/whisper-large-v3/snapshots/xxx
export WAV_INPUT_DIR=./records

python -m whisper_asr wav
```

Options: `--input-dir --output-dir --verbose`

## Configuration

Everything via env vars (see `src/whisper_asr/config.py` for full list).

| Env | Default | Description |
|-----|---------|-------------|
| `WHISPER_MODEL_PATH` | `""` | Local model path (falls back to `WHISPER_MODEL_ID`) |
| `WHISPER_MODEL_ID` | `openai/whisper-large-v3` | HuggingFace model ID |
| `ASR_DEVICE` | `cuda:0` | `cuda:0` or `cpu` |
| `ASR_DTYPE` | `float16` | `float16` or `float32` |
| `OLLAMA_HOST` | `localhost` | Ollama server host |
| `OLLAMA_MODEL` | `gemma-4-E4b-it:latest` | LLM for correction |
| `LLM_CORRECTION` | `true` | Enable/disable LLM post-processing |
| `PARQUET_PATH` | `""` | Input parquet path |
| `PARQUET_ROWS` | `100` | Rows to process |
| `PARQUET_OUTPUT` | `output.csv` | Results CSV |
| `WAV_INPUT_DIR` | `""` | Input directory with .wav files |
| `WAV_OUTPUT_DIR` | `transcripts` | Output directory for .txt files |

## Structure

```
src/whisper_asr/
├── __init__.py
├── __main__.py      # CLI entry point
├── config.py         # env-var configuration
├── audio.py          # load audio (bytes/file), resample
├── transcribe.py     # Whisper pipeline
├── correct.py        # LLM spelling correction
├── evaluate.py       # WER, CER, similarity metrics
└── text.py           # Persian normalization
```

## Dependencies

`torch`, `transformers`, `librosa`, `soundfile`, `pandas`, `jiwer`, `rapidfuzz`, `sentence-transformers`, `langchain-ollama`, `hazm`
