import os
import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# 1. Initial setup
main_path =  "/home/bahmanabadi/Projects/asr/"
model_path = main_path + "models/hub/models--openai--whisper-large-v3/snapshots/06f233fe06e710322aca913c1bc4249a0d71fce1"
input_dir = main_path + "records"
output_dir = "./transcripts"

os.makedirs(output_dir, exist_ok=True)

# 2. Device and dtype setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# 3. Load model and processor
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_path,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_path)

# 4. Create ASR pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# 5. Process all WAV files
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".wav"):
        file_path = os.path.join(input_dir, filename)

        # Load audio without ffmpeg
        audio_array, sampling_rate = librosa.load(file_path, sr=16000, mono=True)

        # Run transcription
        result = pipe(
            {"array": audio_array, "sampling_rate": sampling_rate},
            generate_kwargs={"language": "persian"}
        )

        text = result["text"].strip()

        print(f"File: {filename}")
        print(f"Text: {text}")
        print("-" * 50)

        # Save transcription
        output_file = os.path.join(
            output_dir,
            f"{os.path.splitext(filename)[0]}_transcript.txt"
        )
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)

print("Processing complete.")