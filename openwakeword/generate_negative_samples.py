import os
import random
import uuid

import librosa
import soundfile as sf
from tqdm import tqdm

SAMPLE_RATE = 16000

def list_all_wav_files(base_path):
    wav_files = []
    for root, _, files in os.walk(base_path):
        for f in files:
            if f.lower().endswith(".wav"):
                wav_files.append(os.path.join(root, f))
    return wav_files

def save_clip(clip, directory):
    file_id = uuid.uuid4().hex + ".wav"
    sf.write(os.path.join(directory, file_id), clip, SAMPLE_RATE)

def extract_clip(file_path, duration, sr, output_dir):
    try:
        y, _ = librosa.load(file_path, sr=sr, mono=True)
        total_samples = int(duration * sr)
        if len(y) < total_samples:
            return None
        start = random.randint(0, len(y) - total_samples)
        clip = y[start:start + total_samples]
        save_clip(clip, output_dir)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def generate_negative_samples(dataset: str, output_dir: str, number: int, duration: int):
    print("🔍 Listing all wav files in DATASET...")
    all_files = list_all_wav_files(dataset)
    random.shuffle(all_files)

    print(f"🎧 Extracting samples...")
    os.makedirs(output_dir, exist_ok=True)
    for i in tqdm(range(number)):
        if i < len(all_files):
            file_path = all_files[i]
        else:
            file_path = random.choices(all_files, k=1)[0]
        extract_clip(file_path, duration, SAMPLE_RATE, output_dir)
    print("Generate negative samples finished")
