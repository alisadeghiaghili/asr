# Persian ASR Pipeline for IME Call Center

> **Goal:** Automatically transcribe and analyze Persian call-center recordings from the **Iran Mercantile Exchange (IME)** to extract structured information, detect customer issues, and evaluate agent performance — enabling data-driven insights from voice interactions.

---

## 1. Problem Statement

The Iran Mercantile Exchange handles a high volume of customer support calls. These recordings contain valuable information — customer complaints, agent ratings, process feedback, and resolved/unresolved issues — but are locked in audio format and cannot be efficiently analyzed at scale.

**Current challenges:**

- Manual transcription is time-consuming and inconsistent
- Customer problems are buried in unstructured audio
- Agent performance tracking relies on subjective manual review
- No systematic way to categorize or track recurring issues

**This project solves these problems** by providing an automated pipeline that converts raw call recordings into structured, actionable data.

---

## 2. Goals & Objectives

| Goal | Description |
|------|-------------|
| **Automated Transcription** | Convert Persian call recordings to text with high accuracy |
| **Error Correction** | Fix ASR errors (spelling, grammar, punctuation) using LLM post-processing |
| **Information Extraction** | Extract call ID, transcript, agent rating, process helpfulness, problems, and general topics from each call |
| **Quality Evaluation** | Benchmark multiple ASR models using WER, CER, similarity metrics to find the best configuration |
| **Structured Output** | Produce CSV/Parquet datasets ready for downstream analytics and reporting |

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT LAYER                           │
│  Call Recordings (.wav) │ Parquet Datasets │ Raw Audio   │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                 AUDIO PREPROCESSING                      │
│  • Sample rate conversion (→ 16kHz mono)                │
│  • Silence detection & chunking                         │
│  • Format normalization (WAV/MP3 → standardized)        │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                  ASR ENGINE                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Whisper Small │  │Whisper Large │  │ Gemma 4      │  │
│  │ (HF Snapshot) │  │ v3 Turbo     │  │ Multimodal   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              LLM POST-PROCESSING (Ollama)               │
│  • Spelling & grammar correction                        │
│  • Call-center information extraction                   │
│  • Structured JSON output per call                      │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Models: Gemma, Qwen, GPT-OSS                   │    │
│  │  Output: { callID, callTranscript, agentRate,    │    │
│  │           processHelpfulness, problems[],        │    │
│  │           generalTopics[] }                      │    │
│  └─────────────────────────────────────────────────┘    │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│               EVALUATION & OUTPUT                        │
│  • FuzzyWuzzy / Levenshtein similarity                  │
│  • Semantic similarity (sentence-transformers)           │
│  • WER / CER (jiwer)                                    │
│  • CSV / Parquet / Excel export                         │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Project Structure

```
asr/
│
├── asr.ipynb                    # Main pipeline: Whisper + LLM correction
│                                #   ├─ Whisper Small (HF transformers)
│                                #   ├─ Whisper Medium (.pt via openai-whisper)
│                                #   └─ Whisper Large v3 Turbo (.bin via pywhispercpp)
│
├── gemma-asr.ipynb              # Gemma 4 multimodal ASR (direct audio → text)
│
├── similarity.ipynb             # Evaluation: FuzzyWuzzy + Semantic similarity
│
├── parquet_file_test.ipynb      # Evaluation on parquet datasets (WER/CER)
│
├── README.md                    # This document
│
└── (runtime files)
    ├── records/                 # Input audio files (.wav)
    ├── records_16k/             # Resampled 16kHz audio files
    ├── models/                  # Local model weights
    │   ├── whisper-small/       # Whisper Small (HuggingFace snapshot)
    │   ├── medium.pt            # Whisper Medium (OpenAI .pt)
    │   └── ggml-large-v3-turbo.bin  # Whisper Large v3 Turbo (.bin)
    └── prompt.txt               # LLM correction prompt template
```

---

## 5. ASR Model Comparison

The project evaluates three Whisper variants and one multimodal model. Each has different tradeoffs for the call-center use case.

### 5.1 Whisper Small (HuggingFace Transformers)

| Property | Value |
|----------|-------|
| **Backend** | `transformers.WhisperForConditionalGeneration` |
| **Model Path** | `models/models--openai--whisper-small/snapshots/...` |
| **Inference** | PyTorch, CPU/GPU |
| **Strengths** | Lightweight, fast, works offline |
| **Weaknesses** | Lower accuracy on noisy call-center audio |

### 5.2 Whisper Medium (OpenAI .pt)

| Property | Value |
|----------|-------|
| **Backend** | `whisper.load_model()` (openai-whisper) |
| **Model Path** | `models/medium.pt` |
| **Inference** | Requires `ffmpeg` |
| **Strengths** | Better accuracy than Small |
| **Weaknesses** | Heavier, requires ffmpeg dependency |

### 5.3 Whisper Large v3 Turbo (whisper.cpp)

| Property | Value |
|----------|-------|
| **Backend** | `pywhispercpp` (C++ bindings) |
| **Model Path** | `models/ggml-large-v3-turbo.bin` |
| **Inference** | C++ native, GPU-accelerated |
| **Strengths** | Best accuracy, fast inference, multilingual |
| **Weaknesses** | Large model size (~1.6 GB) |
| **Config** | `language="fa"`, `n_threads=8`, `no_context=True` |

### 5.4 Gemma 4 Multimodal (Direct ASR)

| Property | Value |
|----------|-------|
| **Backend** | `transformers.AutoModelForMultimodalLM` |
| **Model Path** | `models/models--google--gemma-4-E4B-it/snapshots/...` |
| **Approach** | Audio-in → text-out (no separate ASR stage) |
| **Strengths** | End-to-end, single model for transcription |
| **Weaknesses** | Requires significant GPU memory |

---

## 6. LLM Post-Processing

Raw ASR output from call-center recordings typically contains:

- Persian spelling errors (ASR misrecognitions)
- Missing punctuation
- Grammar inconsistencies
- Misheard domain-specific terms

### 6.1 Correction Pipeline

Each raw transcription is sent to an LLM via Ollama for correction:

```python
llm = ChatOllama(
    model="gemma-4-E4b-it:latest",  # or qwen2.5:7b, gpt-oss:20b
    base_url="http://localhost:11434",
    temperature=0.0,
)
```

### 6.2 Prompt Design

The prompt instructs the LLM to:

1. Fix spelling and punctuation errors
2. Correct ASR misrecognitions when context makes the intended word clear
3. Preserve the original meaning exactly
4. Output only the corrected text (no explanations)

**Example:**

| Input (Raw ASR) | Output (LLM Corrected) |
|-----------------|----------------------|
| سلام حال شما چطوره امیدوارم خوب باشین | سلام، حال شما چطور است؟ امیدوارم خوب باشید. |
|وقتی خطر را تشخیص می دهد | وقتی خطر را تشخیص می‌دهد. |

### 6.3 Call-Center Information Extraction

Beyond correction, the LLM extracts structured information from each call:

```json
{
  "callID": "N/A",
  "callTranscript": "ریسک بالا، پاداش پایین. از این وضعیت متنفرم.",
  "agentRate": "N/A",
  "processHelpfulness": "N/A",
  "problems": [
    {
      "topic": "سرمایه‌گذاری/ریسک مالی",
      "description": "وضعیت سرمایه‌گذاری با ریسک بالا و پاداش پایین.",
      "resolutionStatus": "N/A",
      "actionTaken": "N/A"
    }
  ],
  "generalTopics": []
}
```

**Extracted fields:**

| Field | Description |
|-------|-------------|
| `callID` | Unique call identifier (if available in audio) |
| `callTranscript` | Corrected Persian transcript |
| `agentRate` | Agent performance rating (if mentioned) |
| `processHelpfulness` | Process helpfulness rating (if mentioned) |
| `problems` | List of customer issues with topic, description, resolution status, and action taken |
| `generalTopics` | General discussion topics that are not problems |

### 6.4 Supported LLM Models

| Model | Use Case |
|-------|----------|
| `gemma-4-E4b-it:latest` | Primary — correction + extraction |
| `gemma4:31b` | Higher accuracy extraction |
| `qwen2.5:7b` | Lightweight alternative |
| `gpt-oss:20b` | Alternative for correction-only |

---

## 7. Evaluation Methodology

To determine the best ASR configuration for IME call-center data, the project uses multiple evaluation approaches.

### 7.1 FuzzyWuzzy / Levenshtein Similarity

Character-level string similarity between raw ASR output and LLM-corrected transcript.

```
Score range: 0–100 (higher = more similar)
```

**Results (120 samples):**

| Range | Count | Interpretation |
|-------|-------|----------------|
| 90–100 | ~50 | Excellent — minimal correction needed |
| 70–89 | ~40 | Good — minor spelling/grammar fixes |
| 50–69 | ~8 | Moderate — noticeable ASR errors |
| < 50 | ~4 | Poor — significant transcription errors |

### 7.2 Semantic Similarity

Embedding-based similarity using `paraphrase-multilingual-MiniLM-L12-v2`:

```
Score range: 0–100 (cosine similarity × 100)
```

This captures meaning preservation even when wording differs.

### 7.3 WER / CER (Parquet Dataset Evaluation)

For datasets with ground-truth transcriptions (e.g., SanayAI podcast dataset):

| Metric | Description |
|--------|-------------|
| **WER** (Word Error Rate) | Percentage of words incorrectly transcribed |
| **CER** (Character Error Rate) | Percentage of characters incorrectly transcribed |

**Results (100 samples, Gemma 4 on SanayAI dataset):**

| Metric | Average |
|--------|---------|
| WER | ~0.35 |
| CER | ~0.15 |
| Exact Match | ~5% |
| Normalized Match | ~15% |

### 7.4 Evaluation Pipeline

```
input_csv (raw ASR + LLM output)
    │
    ├─→ similarity.ipynb
    │     ├─ FuzzyWuzzy Levenshtein
    │     └─ Semantic Similarity (sentence-transformers)
    │
    └─→ parquet_file_test.ipynb
          ├─ WER (jiwer)
          └─ CER (jiwer)
```

---

## 8. Setup & Installation

### 8.1 Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for Gemma 4 / Large v3 Turbo)
- Ollama running locally on port 11434
- ffmpeg (for Whisper Medium .pt model)

### 8.2 Install Dependencies

```bash
pip install openai-whisper librosa langchain-ollama fuzzywuzzy python-Levenshtein
pip install sentence-transformers jiwer pywhispercpp pydub
pip install transformers accelerate torch soundfile numpy pandas openpyxl
```

### 8.3 Download Models

**Whisper models:**
```bash
# Small (HuggingFace snapshot — download via transformers)
# Medium (.pt) — download from OpenAI Whisper releases
# Large v3 Turbo (.bin) — download from whisper.cpp releases
```

**Gemma 4 (HuggingFace):**
```bash
# Download via transformers or huggingface-cli
# Place in models/models--google--gemma-4-E4B-it/snapshots/...
```

### 8.4 Setup Ollama

```bash
# Install Ollama, then:
ollama pull gemma-4-E4b-it:latest
# or
ollama pull qwen2.5:7b
```

### 8.5 Configuration

Update paths in notebooks:

```python
main_path = "/path/to/your/asr/project/"
model_path = main_path + "models/ggml-large-v3-turbo.bin"
audio_folder = main_path + "records/"
output_csv = "./transcriptions.csv"

OLLAMA_HOST = "localhost"
OLLAMA_PORT = "11434"
OLLAMA_MODEL = "gemma-4-E4b-it:latest"
```

---

## 9. Usage

### 9.1 Transcription Pipeline (asr.ipynb)

1. Place `.wav` audio files in `records/`
2. Open `asr.ipynb`
3. Run the cell for your chosen ASR model variant
4. Output saved to CSV with columns: `filename`, `transcription`, `llm_output`

### 9.2 Gemma Direct ASR (gemma-asr.ipynb)

1. Requires Gemma 4 multimodal model locally
2. Feed audio files directly — no separate ASR step
3. Produces transcriptions without Whisper dependency

### 9.3 Evaluation (similarity.ipynb)

1. Requires CSV output from the transcription pipeline
2. Run FuzzyWuzzy/Levenshtein similarity
3. Run semantic similarity
4. Output: CSV with similarity scores per file

### 9.4 Parquet Evaluation (parquet_file_test.ipynb)

1. Place parquet dataset in project directory
2. Run evaluation with WER/CER metrics
3. Output: enriched parquet + Excel report

---

## 10. Audio Preprocessing

### 10.1 Resampling to 16kHz

All audio is resampled to 16kHz mono for consistent ASR input:

```python
audio, sr = librosa.load(audio_path, sr=16000, mono=True)
```

### 10.2 Silence Detection

For long call recordings, silence detection splits audio into meaningful chunks:

```python
from pydub import AudioSegment
from pydub.silence import split_on_silence

audio = AudioSegment.from_wav("call_recording.wav")
chunks = split_on_silence(
    audio,
    min_silence_len=700,
    silence_thresh=-40,
    keep_silence=300
)
```

### 10.3 Format Support

| Format | Support |
|--------|---------|
| WAV (16kHz mono) | Primary — recommended |
| WAV (other rates) | Auto-resampled to 16kHz |
| MP3 | Supported via librosa |
| Parquet (embedded audio) | Supported — auto-detected |

---

## 11. Output Formats

### CSV Output

```
transcriptions_corrected.csv
```

| filename | transcription | llm_output |
|----------|--------------|------------|
| record001.wav | سلام حال شما چطوره | {"callTranscript": "سلام، حال شما چطور است؟", ...} |

Encoding: **UTF-8-SIG** (ensures correct Persian display in Excel)

### Parquet Output

Enriched parquet with columns:

| Column | Description |
|--------|-------------|
| `gemma_transcription` | Model prediction |
| `exact_match` | Boolean — exact match with ground truth |
| `normalized_match` | Boolean — normalized match |
| `wer` | Word Error Rate |
| `cer` | Character Error Rate |

### Excel Export

Available via `openpyxl` for reporting and sharing with non-technical stakeholders.

---

## 12. Tech Stack

| Layer | Tools |
|-------|-------|
| **ASR** | Whisper Small/Medium/Large-v3-Turbo, Gemma 4 Multimodal |
| **LLM** | Gemma, Qwen, GPT-OSS (via Ollama) |
| **LLM Framework** | LangChain (langchain-ollama) |
| **Audio Processing** | Librosa, SoundFile, PyDub |
| **Deep Learning** | PyTorch, Transformers (HuggingFace) |
| **Evaluation** | jiwer (WER/CER), FuzzyWuzzy, sentence-transformers |
| **Data** | Pandas, Parquet, CSV, OpenPyXL |

---

## 13. Key Findings

1. **Whisper Large v3 Turbo** (via whisper.cpp) provides the best balance of accuracy and speed for Persian call-center audio
2. **Gemma 4 multimodal** can transcribe directly without a separate ASR model, but requires significant GPU memory
3. **LLM post-processing** consistently improves raw ASR output — fixing spelling, adding punctuation, and correcting misrecognitions
4. **Semantic similarity** scores are generally higher than FuzzyWuzzy scores, indicating the LLM preserves meaning even when wording changes
5. **Call-center extraction** (problems, topics, ratings) is feasible from corrected transcripts, though accuracy depends on audio quality

---

## 14. Future Work

| Area | Description |
|------|-------------|
| **Real-time processing** | Stream transcription for live call monitoring |
| **Speaker diarization** | Distinguish agent vs. customer voice |
| **Sentiment analysis** | Detect customer emotion from transcript |
| **Issue tracking** | Aggregate problems over time for trend analysis |
| **Model fine-tuning** | Fine-tune Whisper on IME-specific vocabulary |
| **API deployment** | Expose pipeline as REST API for integration with call-center systems |
| **Dashboard** | Build analytics dashboard for call-center management |

---

## 15. Author

**Melika Bahman-Abadi**

---

## 16. References

- [OpenAI Whisper](https://github.com/openai/whisper)
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
- [Ollama](https://ollama.com/)
- [LangChain](https://langchain.com/)
- [Librosa](https://librosa.org/)
- [jiwer](https://github.com/jitsi/jiwer)
- [sentence-transformers](https://www.sbert.net/)

---

## 17. License

For research and development use.