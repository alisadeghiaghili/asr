#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install -i https://pypi.devneeds.ir/simple/ fuzzywuzzy


# In[ ]:


pip install -i https://pypi.devneeds.ir/simple/ sentence_transformers


# ## Fuzzywuzzy, Levenshtein Similarity

# In[3]:


import csv
import json
import os

# You may need to install fuzzywuzzy: pip install fuzzywuzzy python-Levenshtein
from fuzzywuzzy import fuzz

# =========================
# Input & Output Paths
# =========================
input_csv = "./ggml-v3-turbo&gemma-4.csv"   # CSV file generated in the previous step
output_csv = "./ggml-v3-turbo&gemma-4-fuzzywuzzy&Levenshtein-similarity.csv"

# =========================
# Read & Process
# =========================
with open(input_csv, "r", encoding="utf-8-sig") as infile, \
     open(output_csv, "w", newline="", encoding="utf-8-sig") as outfile:

    reader = csv.DictReader(infile)   # read with headers
    # Output columns: filename, transcription, llm_text, similarity
    fieldnames = ["filename", "transcription", "llm_text", "similarity"]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        filename = row["filename"]
        transcription = row["transcription"]
        llm_output_str = row["llm_output"]

        # Extract callTranscript from JSON
        callTranscript = ""
        similarity = None

        try:
            llm_data = json.loads(llm_output_str)
            callTranscript = llm_data.get("callTranscript", "")
            if not callTranscript:
                callTranscript = ""   # if key missing or empty
        except Exception as e:
            print(f"Error parsing JSON for {filename}: {e}")
            callTranscript = ""   # fallback to empty string

        # Compute similarity (only if both strings are non-empty)
        if transcription and callTranscript:
            similarity = fuzz.ratio(transcription, callTranscript)   # score 0-100
        elif not transcription and not callTranscript:
            similarity = 100.0   # both empty -> perfect match
        else:
            similarity = 0.0     # one empty -> no similarity

        # Write the row with llm_text column name
        writer.writerow({
            "filename": filename,
            "transcription": transcription,
            "llm_text": callTranscript,
            "similarity": similarity
        })

        # Console report
        print(f"{filename}: similarity = {similarity}")

print(f"\nDone! Output saved to: {output_csv}")


# ## Semantic Similarity

# In[4]:


import csv
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# تنظیم endpoint (اختیاری – در صورت نیاز)
os.environ["HF_ENDPOINT"] = "https://huggingface.co"

# =========================
# Input & Output Paths
# =========================
input_csv = "./ggml-v3-turbo&gemma-4.csv"
output_csv = "./ggml-v3-turbo&gemma-4-semantic_similarity.csv"

# =========================
# Define local model path
# =========================
local_model_path = "/home/hossein/Projects/Jupyter/Bahmanabadi/MrsBahmanAbadi/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/e8f8c211226b894fcb81acc59f3b34ba3efd5f42"

# =========================
# Load Local Model (force local only)
# =========================
try:
    model = SentenceTransformer(local_model_path, local_files_only=True)
    print("Local semantic similarity model loaded successfully.")
except Exception as e:
    print(f"Failed to load model from local path: {e}")
    # اگر باز هم خطا می‌دهد، مسیر را چک کنید (ممکن است snapshot ناقص باشد)
    # راه‌حل جایگزین: استفاده از TF-IDF در نظر گرفته شود
    exit()

# =========================
# Read & Process
# =========================
with open(input_csv, "r", encoding="utf-8-sig") as infile, \
     open(output_csv, "w", newline="", encoding="utf-8-sig") as outfile:

    reader = csv.DictReader(infile)
    fieldnames = ["filename", "transcription", "llm_text", "similarity_semantic"]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        filename = row["filename"]
        transcription = row["transcription"]
        llm_output_str = row["llm_output"]

        # Extract callTranscript from JSON
        callTranscript = ""
        try:
            llm_data = json.loads(llm_output_str)
            callTranscript = llm_data.get("callTranscript", "")
            if not callTranscript:
                callTranscript = ""
        except Exception as e:
            print(f"Error parsing JSON for {filename}: {e}")
            callTranscript = ""

        # Compute semantic similarity
        if transcription and callTranscript:
            emb1 = model.encode(transcription, convert_to_tensor=False)
            emb2 = model.encode(callTranscript, convert_to_tensor=False)
            cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            similarity = round(float(cos_sim) * 100, 2)
        elif not transcription and not callTranscript:
            similarity = 100.0
        else:
            similarity = 0.0

        writer.writerow({
            "filename": filename,
            "transcription": transcription,
            "llm_text": callTranscript,
            "similarity_semantic": similarity
        })

        print(f"{filename}: semantic similarity = {similarity}")

print(f"\nDone! Output saved to: {output_csv}")


# In[ ]:




