#!/usr/bin/env python
# coding: utf-8

# In[20]:


pip install -i https://pypi.devneeds.ir/simple/ openpyxl
pip install -i https://pypi.devneeds.ir/simple/ jiwer


# In[ ]:


import io
import os
import tempfile

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch

from jiwer import wer, cer

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText
)

# =====================================================
# CONFIG
# =====================================================

LOCAL_MODEL_DIR = "/home/bahmanabadi/Projects/asr/models/models--google--gemma-4-E4B-it/snapshots/83df0a889143b1dbfc61b591bbc639540fd9ce4c/"

INPUT_PARQUET = "/home/bahmanabadi/Projects/asr/ASRDataset/SanayAIchannelbpodcast/data/train-00000-of-00003.parquet"

OUTPUT_PARQUET = "parquet/SanayAIchannelbpodcast.parquet"

MAX_ROWS = 100
TARGET_SR = 16000

# =====================================================
# TEXT HELPERS
# =====================================================

def normalize_fa(text):

    if text is None:
        return ""

    text = str(text)

    replacements = {
        "ي": "ی",
        "ك": "ک",
        "ة": "ه",
        "ؤ": "و",
        "أ": "ا",
        "إ": "ا",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = text.replace("‌", " ")

    text = " ".join(text.split())

    return text.strip()


def safe_wer(reference, prediction):

    try:
        return wer(reference, prediction)
    except Exception:
        return None


def safe_cer(reference, prediction):

    try:
        return cer(reference, prediction)
    except Exception:
        return None


# =====================================================
# AUDIO HELPERS
# =====================================================

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

    # -----------------------------------------------
    # WAV
    # -----------------------------------------------

    if audio_format == "wav":

        waveform, sample_rate = sf.read(
            io.BytesIO(audio_bytes)
        )

    # -----------------------------------------------
    # MP3
    # -----------------------------------------------

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

    # -----------------------------------------------
    # TO NUMPY
    # -----------------------------------------------

    waveform = np.asarray(waveform)

    # -----------------------------------------------
    # STEREO -> MONO
    # -----------------------------------------------

    if waveform.ndim == 2:

        # librosa => (channels, samples)
        if waveform.shape[0] <= 8:
            waveform = waveform.mean(axis=0)

        # soundfile => (samples, channels)
        else:
            waveform = waveform.mean(axis=1)

    waveform = waveform.astype(np.float32)

    # -----------------------------------------------
    # RESAMPLE TO 16K
    # -----------------------------------------------

    if sample_rate != TARGET_SR:

        waveform = librosa.resample(
            waveform,
            orig_sr=sample_rate,
            target_sr=TARGET_SR
        )

        sample_rate = TARGET_SR

    waveform = waveform.astype(np.float32)

    return waveform, sample_rate, audio_format


# =====================================================
# LOAD MODEL
# =====================================================

print("Loading model...")

model = AutoModelForImageTextToText.from_pretrained(
    LOCAL_MODEL_DIR,
    torch_dtype="auto",
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(
    LOCAL_MODEL_DIR
)

print("Model loaded.")

# =====================================================
# LOAD DATA
# =====================================================

print("Loading parquet...")

df = pd.read_parquet(INPUT_PARQUET)

df = df.head(MAX_ROWS)

print(f"Rows in parquet: {len(df)}")

rows_to_process = min(
    MAX_ROWS,
    len(df)
)

# =====================================================
# OUTPUT ARRAYS
# =====================================================

predictions = []

exact_matches = []
normalized_matches = []

wer_scores = []
cer_scores = []

# =====================================================
# PROCESS
# =====================================================

for idx in range(rows_to_process):

    try:

        row = df.iloc[idx]

        # -------------------------------------------
        # REFERENCE TEXT
        # -------------------------------------------

        if "sentence" in df.columns:

            reference_text = normalize_fa(
                row["sentence"]
            )

        elif "transcription" in df.columns:

            reference_text = normalize_fa(
                row["transcription"]
            )

        else:

            raise ValueError(
                "Neither sentence nor transcription column exists."
            )

        # -------------------------------------------
        # AUDIO
        # -------------------------------------------

        audio_bytes = row["audio"]["bytes"]

        waveform, sample_rate, audio_format = load_audio(
            audio_bytes
        )

        # -------------------------------------------
        # GEMMA PROMPT
        # -------------------------------------------

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """
This audio is Persian speech.

Transcribe it in Persian script.

Rules:
- Output Persian text only.
- Do not use Latin characters.
- Do not translate.
- Do not explain.
- Output only the transcription.
"""
                    },
                    {
                        "type": "audio",
                        "audio": waveform
                    }
                ]
            }
        ]

        # -------------------------------------------
        # TOKENIZE
        # -------------------------------------------

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )

        inputs = {
            k: v.to(model.device)
            if hasattr(v, "to")
            else v
            for k, v in inputs.items()
        }

        # -------------------------------------------
        # GENERATE
        # -------------------------------------------

        outputs = model.generate(
            **inputs,
            max_new_tokens=128
        )

        prediction = processor.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # -------------------------------------------
        # CLEAN OUTPUT
        # -------------------------------------------

        if "model\n" in prediction:
            prediction = prediction.split("model\n")[-1]

        prediction = normalize_fa(
            prediction
        )

        # -------------------------------------------
        # METRICS
        # -------------------------------------------

        exact_match = (
            prediction == reference_text
        )

        normalized_match = (
            normalize_fa(prediction)
            ==
            normalize_fa(reference_text)
        )

        wer_value = safe_wer(
            reference_text,
            prediction
        )

        cer_value = safe_cer(
            reference_text,
            prediction
        )

        # -------------------------------------------
        # STORE
        # -------------------------------------------

        predictions.append(
            prediction
        )

        exact_matches.append(
            exact_match
        )

        normalized_matches.append(
            normalized_match
        )

        wer_scores.append(
            wer_value
        )

        cer_scores.append(
            cer_value
        )

        print(
            f"[{idx+1}/{rows_to_process}] "
            f"format={audio_format} "
            f"sr={sample_rate} "
            f"WER={wer_value:.3f} "
            f"CER={cer_value:.3f}"
        )

    except Exception as e:

        print(
            f"[{idx+1}] ERROR: {e}"
        )

        predictions.append(None)

        exact_matches.append(False)
        normalized_matches.append(False)

        wer_scores.append(None)
        cer_scores.append(None)

# =====================================================
# SAVE RESULTS
# =====================================================

df.loc[
    :rows_to_process-1,
    "gemma_transcription"
] = predictions

df.loc[
    :rows_to_process-1,
    "exact_match"
] = exact_matches

df.loc[
    :rows_to_process-1,
    "normalized_match"
] = normalized_matches

df.loc[
    :rows_to_process-1,
    "wer"
] = wer_scores

df.loc[
    :rows_to_process-1,
    "cer"
] = cer_scores

# =====================================================
# SAVE PARQUET
# =====================================================

df.to_parquet(
    OUTPUT_PARQUET,
    index=False
)

# =====================================================
# SUMMARY
# =====================================================

valid_wer = [
    x for x in wer_scores
    if x is not None
]

valid_cer = [
    x for x in cer_scores
    if x is not None
]

print()
print("=" * 60)

print("Done.")
print(f"Saved: {OUTPUT_PARQUET}")

print()

print(
    f"Exact Match Count: {sum(exact_matches)}"
)

print(
    f"Normalized Match Count: {sum(normalized_matches)}"
)

if len(valid_wer) > 0:

    print(
        f"Average WER: {np.mean(valid_wer):.4f}"
    )

if len(valid_cer) > 0:

    print(
        f"Average CER: {np.mean(valid_cer):.4f}"
    )

print("=" * 60)


# In[ ]:


import pandas as pd
import openpyxl

df = pd.read_parquet("parquet/SanayAIchannelbpodcast.parquet")
df.head()


# In[8]:


report_df = df[
[
"sentence",
"gemma_transcription",
"exact_match",
"normalized_match",
"wer",
"cer"
]
]

report_df.to_excel(
    "parquet/SanayAIchannelbpodcast.xlsx",
    index=False
)


# In[ ]:




