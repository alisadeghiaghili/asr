#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install -i https://pypi.devneeds.ir/simple/ openai-whisper


# In[ ]:


pip install -i https://pypi.devneeds.ir/simple/ langchain_ollama


# ## Snapshot model (small)

# In[ ]:


from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa
import os
import glob
import csv

from langchain_ollama import ChatOllama

# =========================
# Paths
# =========================
main_path = "/home/bahmanabadi/Projects/asr/"
model_path = main_path + "models/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d"
audio_folder = main_path + "records/"
output_csv = "./transcriptions.csv"

# =========================
# Ollama Remote Config
# =========================
OLLAMA_HOST = "localhost"
OLLAMA_PORT = "11434"
OLLAMA_MODEL = "gpt-oss:20b"

# =========================
# Load Whisper
# =========================
processor = WhisperProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(model_path)

# =========================
# Load Ollama LLM
# =========================
base_url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=base_url,
    temperature=0.0,
)

print(f"LLM Loaded: {OLLAMA_MODEL} @ {base_url}")

# =========================
# Audio Files
# =========================
audio_files = glob.glob(os.path.join(audio_folder, "*.wav"))


# =========================
# Spelling Correction
# =========================
def correct_spelling(text):

    prompt = f"""
Task: Correct spelling and punctuation errors in the given text.

Rules:
- Fix spelling mistakes and punctuation only.
- You may rewrite the sentence if necessary to correct those errors.
- Do NOT add any new words that were not present in the original text.
- Do NOT remove important words unless they are clearly duplicated because of an error.
- Preserve the original meaning of the text.
- Keep the same language as the input:
  - If the input text is Persian, the output must be Persian.
  - If the input text is English, the output must be English.
- Output only the corrected text and nothing else.

Text:
{text}

Corrected text:

"""

    try:
        response = llm.invoke(prompt)

        # استخراج متن خروجی
        corrected_text = response.content.strip()

        return corrected_text

    except Exception as e:
        print("LLM Error:", e)
        return text


# =========================
# CSV Output
# =========================
with open(output_csv, "w", newline="", encoding="utf-8-sig") as csvfile:

    writer = csv.writer(csvfile)

    # Header
    writer.writerow([
        "filename",
        "transcription",
        "llm_output"
    ])

    for audio_path in audio_files:

        # -----------------
        # Load Audio
        # -----------------
        audio, sr = librosa.load(
            audio_path,
            sr=16000,
            mono=True
        )

        # -----------------
        # Whisper Inference
        # -----------------
        inputs = processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt"
        )

        with torch.no_grad():
            predicted_ids = model.generate(
                inputs.input_features
            )

        transcription = processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]

        # -----------------
        # LLM Correction
        # -----------------
        llm_output = correct_spelling(transcription)

        # -----------------
        # Save
        # -----------------
        filename = os.path.basename(audio_path)

        writer.writerow([
            filename,
            transcription,
            llm_output
        ])

        # -----------------
        # Print Logs
        # -----------------
        print("=" * 60)
        print("File:", filename)
        print("Transcription:")
        print(transcription)
        print("-" * 30)
        print("LLM Output:")
        print(llm_output)

print("\nDone.")
print(f"CSV saved to: {output_csv}")


# ## PT model (medium)

# In[4]:


pip install -i https://pypi.devneeds.ir/simple/ ffmpeg


# In[ ]:


import whisper
import librosa
import os
import glob
import csv

from langchain_ollama import ChatOllama

# =========================
# Paths
# =========================
main_path = "/home/bahmanabadi/Projects/asr/"
model_path = main_path + "models/medium.pt"
audio_folder = main_path + "records/"
output_csv = "./transcriptions_medium.csv"

# =========================
# Ollama Remote Config
# =========================
OLLAMA_HOST = "localhost"
OLLAMA_PORT = "11434"
OLLAMA_MODEL = "qwen2.5:7b" # gpt-oss:20b

# =========================
# Load Whisper
# =========================
model = whisper.load_model(model_path)

print(f"Whisper Model Loaded: {model_path}")

# =========================
# Load Ollama
# =========================
base_url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=base_url,
    temperature=0.0,
)

print(f"LLM Loaded: {OLLAMA_MODEL} @ {base_url}")

# =========================
# Audio Files
# =========================
audio_files = glob.glob(os.path.join(audio_folder, "*.wav"))

print(f"Found {len(audio_files)} audio files.")

# =========================
# Spelling Correction
# =========================
def correct_spelling(text):

    prompt = f"""
Task: Correct spelling and punctuation errors in the given text.

Rules:
- Fix spelling mistakes and punctuation only.
- You may rewrite the sentence if necessary to correct those errors.
- Do NOT add any new words that were not present in the original text.
- Do NOT remove important words unless they are clearly duplicated because of an error.
- Preserve the original meaning of the text.
- Keep the same language as the input.
- Output only the corrected text.

Text:
{text}

Corrected text:
"""

    try:
        response = llm.invoke(prompt)
        return response.content.strip()

    except Exception as e:
        print("LLM Error:", e)
        return text


# =========================
# CSV Output
# =========================
with open(output_csv, "w", newline="", encoding="utf-8-sig") as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow([
        "filename",
        "transcription",
        "llm_output"
    ])

    for audio_path in audio_files:

        try:

            # -----------------
            # Load Audio (NO FFMPEG)
            # -----------------
            audio, sr = librosa.load(
                audio_path,
                sr=16000,
                mono=True
            )

            # -----------------
            # Whisper Inference
            # -----------------
            result = model.transcribe(audio)

            transcription = result["text"].strip()

            # -----------------
            # LLM Correction
            # -----------------
            llm_output = correct_spelling(transcription)

            # -----------------
            # Save
            # -----------------
            filename = os.path.basename(audio_path)

            writer.writerow([
                filename,
                transcription,
                llm_output
            ])

            # -----------------
            # Logs
            # -----------------
            print("=" * 60)
            print("File:", filename)

            print("\nTranscription:")
            print(transcription)

            print("\nLLM Output:")
            print(llm_output)

        except Exception as e:

            print("=" * 60)
            print("Error processing file:", audio_path)
            print("Error:", e)

print("\nDone.")
print(f"CSV saved to: {output_csv}")


# # Bin models

# In[23]:


get_ipython().system('pwd')


# In[ ]:


pip install -i https://pypi.devneeds.ir/simple/ llama-cpp-python


# In[ ]:


get_ipython().system('pip install -i https://pypi.devneeds.ir/simple/ pywhispercpp')


# ## Generate 16khz

# In[ ]:


import os
import glob
import librosa
import soundfile as sf


# =========================
# Paths
# =========================
main_path = "/home/bahmanabadi/Projects/asr/"
audio_folder = main_path + "records/"
output_folder = "/home/hossein/Projects/Jupyter/Bahmanabadi/MrsBahmanAbadi/records_16k/"

# ایجاد پوشه خروجی اگر وجود نداشته باشد
os.makedirs(output_folder, exist_ok=True)


# =========================
# Audio Files
# =========================
audio_files = glob.glob(os.path.join(audio_folder, "*.wav"))

print(f"Found {len(audio_files)} audio files.")


# =========================
# Audio Conversion
# =========================
def convert_to_16k(audio_path, output_folder):
    """
    تبدیل فایل صوتی به نرخ نمونه‌برداری 16kHz
    """
    # بارگذاری و تبدیل نرخ نمونه‌برداری
    audio, sr = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )

    # نام فایل
    filename = os.path.basename(audio_path)
    name, ext = os.path.splitext(filename)

    # مسیر ذخیره
    output_path = os.path.join(output_folder, f"{name}_16k.wav")

    # ذخیره فایل
    sf.write(
        output_path,
        audio,
        16000,
        format="WAV",
        subtype="PCM_16"
    )

    print(f"Converted: {filename} -> {output_path}")

    return output_path


# =========================
# Process All Files
# =========================
for audio_path in audio_files:
    try:
        convert_to_16k(audio_path, output_folder)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")


print("\nDone.")
print(f"Converted files saved to: {output_folder}")


# ## Speech to Text

# In[ ]:


import os
import glob
import csv

from pywhispercpp.model import Model
from langchain_ollama import ChatOllama


# =========================
# Paths
# =========================
main_path = "/home/bahmanabadi/Projects/asr/"
model_path = main_path + "models/ggml-large-v3-turbo.bin"
audio_folder ="/home/hossein/Projects/Jupyter/Bahmanabadi/MrsBahmanAbadi/records_16k/"  # 16KHZ files
output_csv = "./transcriptions_largev3.csv"


# =========================
# Ollama Config
# =========================
OLLAMA_HOST = "localhost"
OLLAMA_PORT = "11434"
OLLAMA_MODEL = "gemma-4-E4b-it:latest"


# =========================
# Load Whisper.cpp model
# =========================
model = Model(model_path)

print(f"Whisper.cpp Model Loaded: {model_path}")


# =========================
# Load Ollama
# =========================
base_url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=base_url,
    temperature=0.0,
)

print(f"LLM Loaded: {OLLAMA_MODEL} @ {base_url}")


# =========================
# Audio Files
# =========================
audio_files = glob.glob(os.path.join(audio_folder, "*.wav"))

print(f"Found {len(audio_files)} audio files.")


# =========================
# Spelling Correction
# =========================
def correct_spelling(text):

    prompt = f"""
Task: Correct spelling and punctuation errors in the given text.

Rules:
- Fix spelling mistakes and punctuation only.
- Do NOT add new words.
- Preserve meaning.
- Keep the same language.
- Output only corrected text.

Text:
{text}

Corrected text:
"""

    try:
        response = llm.invoke(prompt)
        return response.content.strip()

    except Exception as e:
        print("LLM Error:", e)
        return text


# =========================
# CSV Output
# =========================
with open(output_csv, "w", newline="", encoding="utf-8-sig") as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow([
        "filename",
        "transcription",
        "llm_output"
    ])

    for audio_path in audio_files:

        try:

            # -----------------
            # Whisper.cpp inference
            # -----------------
            segments = model.transcribe(
                audio_path,
                language="fa",
                n_threads=8,
                no_context=True)

            transcription = " ".join(seg.text for seg in segments).strip()

            # -----------------
            # LLM correction
            # -----------------
            llm_output = correct_spelling(transcription)

            # -----------------
            # Save CSV
            # -----------------
            filename = os.path.basename(audio_path)

            writer.writerow([
                filename,
                transcription,
                llm_output
            ])

            # -----------------
            # Logs
            # -----------------
            print("=" * 60)
            print("File:", filename)

            print("\nTranscription:")
            print(transcription)

            print("\nLLM Output:")
            print(llm_output)

        except Exception as e:

            print("=" * 60)
            print("Error processing file:", audio_path)
            print("Error:", e)


print("\nDone.")
print(f"CSV saved to: {output_csv}")


# # gemma-4-E4b-it:latest & gemma4:31b models

# ## Silent Detection

# In[35]:


pip install -i https://pypi.devneeds.ir/simple/ pydub


# In[10]:


from pydub import AudioSegment
import os
from pydub.silence import split_on_silence

audio = AudioSegment.from_wav("/home/hossein/Projects/Jupyter/Bahmanabadi/MrsBahmanAbadi/records_16k/telephoni.wav")

chunks = split_on_silence(
    audio,
    min_silence_len=700,   
    silence_thresh=-40,
    keep_silence=300
)

for i, chunk in enumerate(chunks):
    chunk.export(f"chunk_{i}.wav", format="wav")


# ## gemma-4-E4b-it:latest & LLM

# In[ ]:


import glob
import csv
import os

from pywhispercpp.model import Model
from langchain_ollama import ChatOllama


# =========================
# Paths
# =========================
main_path = "/home/bahmanabadi/Projects/asr/"
model_path = main_path + "models/ggml-large-v3-turbo.bin"

audio_folder = "/home/hossein/Projects/Jupyter/Bahmanabadi/MrsBahmanAbadi/records_16k/"
output_csv = "./ggml-v3-turbo&gemma-4-E4b-it:latest_120_test.csv"


# =========================
# Ollama Config
# =========================
OLLAMA_HOST = "localhost"
OLLAMA_PORT = "11434"
OLLAMA_MODEL = "gemma-4-E4b-it:latest"


# =========================
# Load Whisper.cpp
# =========================
model = Model(model_path)

print(f"Whisper.cpp Model Loaded: {model_path}")


# =========================
# Load Ollama LLM
# =========================
base_url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=base_url,
    temperature=0.0,
)

print(f"LLM Loaded: {OLLAMA_MODEL} @ {base_url}")


# =========================
# Get Audio Files
# =========================
audio_files = glob.glob(os.path.join(audio_folder, "*.wav"))

print(f"Found {len(audio_files)} audio files.")


# =========================
# Spelling Correction with Gemma
# =========================
def correct_spelling(text):

    prompt = f"""
    You are a call-center information extraction system.
    
    The conversation below is an automatic speech recognition (ASR) transcript and may contain spelling mistakes, missing punctuation, or recognition errors.
    
    Step 1 — Transcript Editing:
    First clean and correct the transcript:
    - Fix Persian spelling and grammar errors.
    - Correct obvious ASR mistakes when the intended word is clear from context.
    - Add proper punctuation if needed.
    - Keep the original meaning exactly the same.
    - The corrected version must be natural Persian.
    
    Step 2 — Information Extraction:
    Use the corrected transcript to extract the requested information.
    
    Important rules:
    - All extracted values MUST be written in Persian (Farsi).
    - If the original text is not Persian, translate the extracted content to Persian.
    - Be precise and concise.
    - Return structured information for each field.
    
    Fields to extract:
    - Call ID (callID)
    - Full corrected transcript in Persian (callTranscript)
    - Agent performance rating in Persian (agentRate) if mentioned
    - Process helpfulness rating in Persian (processHelpfulness) if mentioned
    - List of problems (each problem must include: topic, description, resolution status, action taken — all in Persian)
    - General topics discussed that are not problems (in Persian)
    
    Conversation:
    {text}
    """


    try:
        response = llm.invoke(prompt)
        return response.content.strip()

    except Exception as e:
        print("LLM Error:", e)
        return text


# =========================
# Process Audio Files
# =========================
with open(output_csv, "w", newline="", encoding="utf-8-sig") as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow([
        "filename",
        "transcription",
        "llm_output"
    ])

    for audio_path in audio_files:
        print("****")

        try:

            # -----------------
            # Whisper Transcription
            # -----------------
            segments = model.transcribe(
                audio_path,
                language="fa",
                n_threads=8,
                no_context=True
            )

            transcription = " ".join(seg.text for seg in segments).strip()


            # -----------------
            # LLM Spelling Correction
            # -----------------
            llm_output = correct_spelling(transcription)


            # -----------------
            # Save to CSV
            # -----------------
            filename = os.path.basename(audio_path)

            writer.writerow([
                filename,
                transcription,
                llm_output
            ])


            # -----------------
            # Console Logs
            # -----------------
            print("=" * 60)
            print("File:", filename)

            print("\nTranscription:")
            print(transcription)

            print("\nLLM Output:")
            print(llm_output)

        except Exception as e:

            print("=" * 60)
            print("Error processing file:", audio_path)
            print("Error:", e)


print("\nDone.")
print(f"CSV saved to: {output_csv}")


# ## json output for llm, gemma-4-E4b-it:latest & LLM

# In[2]:


import glob
import csv
import os
import json

from pywhispercpp.model import Model
from langchain_ollama import ChatOllama

# =========================
# Paths
# =========================
main_path = "/home/bahmanabadi/Projects/asr/"
model_path = main_path + "models/ggml-large-v3-turbo.bin"

audio_folder = "/home/hossein/Projects/Jupyter/Bahmanabadi/MrsBahmanAbadi/records_16k/"
output_csv = "./ggml-v3-turbo&gemma-4.csv"

# =========================
# Ollama Config
# =========================
OLLAMA_HOST = "localhost"
OLLAMA_PORT = "11434"
OLLAMA_MODEL = "gemma-4-E4b-it:latest"

# =========================
# Load Whisper.cpp
# =========================
model = Model(model_path)
print(f"Whisper.cpp Model Loaded: {model_path}")

# =========================
# Load Ollama LLM
# =========================
base_url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=base_url,
    temperature=0.0,
)
print(f"LLM Loaded: {OLLAMA_MODEL} @ {base_url}")

# =========================
# Get Audio Files
# =========================
audio_files = glob.glob(os.path.join(audio_folder, "*.wav"))
print(f"Found {len(audio_files)} audio files.")

# =========================
# Spelling Correction & Extraction (JSON output)
# =========================
def correct_spelling(text):
    # Fixed set of keys for every call
    fixed_keys = {
        "callID": "",
        "callTranscript": "",
        "agentRate": "",
        "processHelpfulness": "",
        "problems": [],          # list of objects with keys: topic, description, resolutionStatus, actionTaken
        "generalTopics": []     # list of strings
    }

    prompt = f"""
        You are a call-center information extraction system.
        
        The conversation below is an automatic speech recognition (ASR) transcript. It may contain spelling mistakes, missing punctuation, or recognition errors.
        
        Step 1 — Transcript Editing:
        First clean and correct the transcript:
        - Fix spelling and grammar errors (Persian or English, depending on the language).
        - Correct obvious ASR mistakes when the intended word is clear from context.
        - Add proper punctuation if needed.
        - Keep the original meaning exactly the same.
        - The corrected version must be natural in the original language (if the input is Persian, output Persian; if English, output English).
        
        Step 2 — Information Extraction:
        Use the corrected transcript to extract the requested information.
        
        Important rules:
        - All extracted values MUST be written in the **same language as the original conversation** (Persian or English).
        - If the original text is not Persian, translate the extracted content to the original language (English). If the original text is Persian, output in Persian.
        - Be precise and concise.
        - Return a **valid JSON object** with the following keys (and no other text outside the JSON):
            - "callID": string (use "N/A" if not found)
            - "callTranscript": string (the corrected transcript, in the original language)
            - "agentRate": string (e.g., "خوب" / "bad" – use the original language; "N/A" if not mentioned)
            - "processHelpfulness": string (e.g., "کمک‌کننده" / "helpful" – in original language; "N/A" if not mentioned)
            - "problems": list of objects, each with keys: "topic" (string), "description" (string), "resolutionStatus" (string), "actionTaken" (string). All values in original language. If no problems, use empty list.
            - "generalTopics": list of strings (topics discussed that are not problems). All in original language. If none, empty list.
        
        Conversation:
        {text}
    """

    try:
        response = llm.invoke(prompt)
        raw_output = response.content.strip()

        # Try to parse JSON
        # Sometimes the model wraps JSON in markdown code blocks, handle that
        if raw_output.startswith("```"):
            # Extract content between ```json and ```
            import re
            match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw_output)
            if match:
                raw_output = match.group(1).strip()
        
        parsed = json.loads(raw_output)
        
        # Validate that all expected keys exist
        for key in fixed_keys:
            if key not in parsed:
                parsed[key] = fixed_keys[key]
        
        # Return JSON string (pretty or compact – we use compact to keep CSV clean)
        return json.dumps(parsed, ensure_ascii=False)

    except Exception as e:
        print("LLM Error or JSON parse error:", e)
        print("Raw output:", raw_output if 'raw_output' in locals() else "N/A")
        # Return a fallback JSON string with empty values
        fallback = {
            "callID": "ERROR",
            "callTranscript": text,
            "agentRate": "N/A",
            "processHelpfulness": "N/A",
            "problems": [],
            "generalTopics": []
        }
        return json.dumps(fallback, ensure_ascii=False)

# =========================
# Process Audio Files
# =========================
with open(output_csv, "w", newline="", encoding="utf-8-sig") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "filename",
        "transcription",
        "llm_output"      # This column will contain JSON string
    ])

    for audio_path in audio_files:
        print("****")
        try:
            # -----------------
            # Whisper Transcription
            # -----------------
            segments = model.transcribe(
                audio_path,
                language="fa",
                n_threads=8,
                no_context=True
            )
            transcription = " ".join(seg.text for seg in segments).strip()

            # -----------------
            # LLM Spelling Correction & Extraction
            # -----------------
            llm_output_json = correct_spelling(transcription)   # returns a JSON string

            # -----------------
            # Save to CSV
            # -----------------
            filename = os.path.basename(audio_path)
            writer.writerow([
                filename,
                transcription,
                llm_output_json   # directly write the JSON string
            ])

            # -----------------
            # Console Logs
            # -----------------
            print("=" * 60)
            print("File:", filename)
            print("\nTranscription:")
            print(transcription)
            print("\nLLM Output (JSON):")
            print(llm_output_json)

        except Exception as e:
            print("=" * 60)
            print("Error processing file:", audio_path)
            print("Error:", e)

print("\nDone.")
print(f"CSV saved to: {output_csv}")


# ## LangChain

# In[ ]:




