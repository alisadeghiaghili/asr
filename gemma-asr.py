#!/usr/bin/env python
# coding: utf-8

# In[5]:


pip install -i https://pypi.devneeds.ir/simple/ --upgrade transformers accelerate


# In[10]:


LOCAL_MODEL_DIR = "/home/bahmanabadi/Projects/asr/models/models--google--gemma-4-E4B-it/snapshots/83df0a889143b1dbfc61b591bbc639540fd9ce4c/"

from transformers import AutoProcessor, AutoModelForMultimodalLM

model = AutoModelForMultimodalLM.from_pretrained(LOCAL_MODEL_DIR, dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(LOCAL_MODEL_DIR)

RESOURCE_URL_PREFIX = "/home/hossein/Projects/Jupyter/Bahmanabadi/MrsBahmanAbadi/records_16k/"

messages = [
{
"role": "user",
"content": [
{"type": "text", "text": "Transcribe the following speech segment in its original language. Follow these specific instructions for formatting the answer:\n* Only output the transcription, with no newlines.\n* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three."},
{"type": "audio", "audio": f"{RESOURCE_URL_PREFIX}record007.wav"},
]
}
]

input_ids = processor.apply_chat_template(
messages,
add_generation_prompt=True,
tokenize=True, return_dict=True,
return_tensors="pt",
enable_thinking = False,
)
input_ids = input_ids.to(model.device, dtype=model.dtype)

outputs = model.generate(**input_ids, max_new_tokens=64)

text = processor.batch_decode(
outputs,
skip_special_tokens=False,
clean_up_tokenization_spaces=False
)
print(text[0])


# In[ ]:




