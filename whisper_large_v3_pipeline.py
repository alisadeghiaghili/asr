import io
import os
import tempfile

import librosa
import pandas as pd
import soundfile as sf
import torch

from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline
)

from langchain_ollama import ChatOllama

from rapidfuzz.distance import Levenshtein
from jiwer import wer, cer

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from hazm import Normalizer


# ==========================================================
# Config
# ==========================================================

main_path = "/home/bahmanabadi/Projects/asr/"

model_path = (
    main_path
    + "models/hub/models--openai--whisper-large-v3/"
    + "snapshots/06f233fe06e710322aca913c1bc4249a0d71fce1"
)

parquet_file = (
    "/home/bahmanabadi/Projects/asr/"
    "ASRDataset/SanayAIbplus_podcast/data/"
    "train-00000-of-00193.parquet"
)

output_csv = "SanayAIbplus_podcast_whisper_largev3.csv"

rows_count = 100


# ==========================================================
# Ollama
# ==========================================================

OLLAMA_HOST = "localhost"
OLLAMA_PORT = "11434"
OLLAMA_MODEL = "gemma-4-E4b-it:latest"

base_url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=base_url,
    temperature=0.0,
)

print(f"LLM Loaded: {OLLAMA_MODEL}")


# ==========================================================
# Semantic Model
# ==========================================================

semantic_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

print("Semantic model loaded")


# ==========================================================
# Normalizer
# ==========================================================

normalizer = Normalizer()


# ==========================================================
# Audio Utils
# ==========================================================

def detect_audio_format(audio_bytes):

    if audio_bytes[:4] == b"RIFF":
        return "wav"

    if audio_bytes[:3] == b"ID3":
        return "mp3"

    if audio_bytes[:2] == b"\xff\xfb":
        return "mp3"

    return "unknown"


def load_audio(audio_bytes):

    audio_format = detect_audio_format(audio_bytes)

    if audio_format == "wav":

        waveform, sample_rate = sf.read(
            io.BytesIO(audio_bytes)
        )

    else:

        with tempfile.NamedTemporaryFile(
            suffix=".mp3",
            delete=False
        ) as tmp:

            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:

            waveform, sample_rate = librosa.load(
                tmp_path,
                sr=None,
                mono=False
            )

        finally:

            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return waveform, sample_rate


# ==========================================================
# Text Cleanup
# ==========================================================

def normalize_text(text):

    text = normalizer.normalize(str(text))

    return text.strip()


# ==========================================================
# LLM Correction
# ==========================================================

def correct_text(text):

    prompt = f"""
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

    try:

        response = llm.invoke(prompt)

        return response.content.strip()

    except Exception as e:

        print("LLM Error:", e)

        return text


# ==========================================================
# Similarity Metrics
# ==========================================================

def calculate_semantic_similarity(
    reference,
    prediction
):

    emb1 = semantic_model.encode(
        [reference]
    )

    emb2 = semantic_model.encode(
        [prediction]
    )

    score = cosine_similarity(
        emb1,
        emb2
    )[0][0]

    return float(score)


def calculate_levenshtein_similarity(
    reference,
    prediction
):

    distance = Levenshtein.distance(
        reference,
        prediction
    )

    max_len = max(
        len(reference),
        len(prediction)
    )

    if max_len == 0:
        return 1.0

    return 1 - (distance / max_len)


# ==========================================================
# Device
# ==========================================================

device = (
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

torch_dtype = (
    torch.float16
    if torch.cuda.is_available()
    else torch.float32
)

print("Device:", device)


# ==========================================================
# Whisper
# ==========================================================

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_path,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True
)

model.to(device)

processor = AutoProcessor.from_pretrained(
    model_path
)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

print("Whisper loaded")


# ==========================================================
# Read Parquet
# ==========================================================

df = pd.read_parquet(parquet_file)

df = df.head(rows_count)

results = []

print(f"Rows: {len(df)}")


# ==========================================================
# Process
# ==========================================================

for idx, row in df.iterrows():

    try:

        audio_bytes = row["audio"]["bytes"]

        waveform, sample_rate = load_audio(
            audio_bytes
        )

        # ------------------------------------
        # Stereo -> Mono
        # ------------------------------------

        if len(waveform.shape) > 1:

            if waveform.shape[0] < waveform.shape[1]:
                waveform = waveform.mean(axis=0)
            else:
                waveform = waveform.mean(axis=1)

        waveform = waveform.astype("float32")

        # ------------------------------------
        # Resample
        # ------------------------------------

        if sample_rate != 16000:

            waveform = librosa.resample(
                waveform,
                orig_sr=sample_rate,
                target_sr=16000
            )

            sample_rate = 16000

        # ------------------------------------
        # Whisper
        # ------------------------------------

        result = pipe(
            {
                "array": waveform,
                "sampling_rate": sample_rate
            },
            generate_kwargs={
                "language": "persian",
                "task": "transcribe"
            }
        )

        whisper_prediction = (
            result["text"]
            .strip()
        )

        # ------------------------------------
        # LLM Cleanup
        # ------------------------------------

        llm_prediction = correct_text(
            whisper_prediction
        )

        # ------------------------------------
        # Metrics
        # ------------------------------------

        reference = normalize_text(
            row["sentence"]
        )

        prediction = normalize_text(
            whisper_prediction
        )

        duration_s = (
            float(row["duration_ms"]) / 1000.0
        )

        semantic_score = (
            calculate_semantic_similarity(
                reference,
                prediction
            )
        )

        levenshtein_score = (
            calculate_levenshtein_similarity(
                reference,
                prediction
            )
        )

        wer_score = wer(
            reference,
            prediction
        )

        cer_score = cer(
            reference,
            prediction
        )

        # ------------------------------------
        # Save Row
        # ------------------------------------

        results.append({

            "row_id":
                idx,

            "sentence":
                reference,

            "whisper_prediction":
                whisper_prediction,

            "llm_prediction":
                llm_prediction,

            "duration_s":
                duration_s,

            "semantic_similarity":
                semantic_score,

            "levenshtein_similarity":
                levenshtein_score,

            "wer":
                wer_score,

            "cer":
                cer_score
        })

        # ------------------------------------
        # Logs
        # ------------------------------------

        print(
            f"[{idx + 1}/{len(df)}]"
        )

        print("REFERENCE:")
        print(reference)

        print()

        print("WHISPER:")
        print(whisper_prediction)

        print()

        print("LLM:")
        print(llm_prediction)

        print()

        print(
            f"WER={wer_score:.4f} | "
            f"CER={cer_score:.4f} | "
            f"SEM={semantic_score:.4f}"
        )

        print("=" * 100)

    except Exception as e:

        print(
            f"Error row {idx}: {e}"
        )


# ==========================================================
# Save CSV
# ==========================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    output_csv,
    index=False,
    encoding="utf-8-sig"
)

print()
print("Done.")
print(f"CSV saved: {output_csv}")