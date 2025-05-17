import os
import random
import uuid

import librosa
import soundfile as sf
from tqdm import tqdm

DATASET_ROOT = "/Users/mengli/Downloads/musan"
DURATION = 2.0
SAMPLE_RATE = 16000
TRAIN_DIR = "resources/navigate_train"
TEST_DIR = "resources/navigate_test"
SAMPLES_PER_FILE = 2
NAVIGATE_TRAIN_SAMPLES = 10000
NAVIGATE_TEST_SAMPLES = 2000

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

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

print("🔍 Listing all wav files in DATASET...")
all_files = list_all_wav_files(DATASET_ROOT)
random.shuffle(all_files)

clips = []
print(f"🎧 Extracting samples...")
for i in tqdm(range(NAVIGATE_TRAIN_SAMPLES + NAVIGATE_TEST_SAMPLES)):
    output_dir = TRAIN_DIR if i <= NAVIGATE_TRAIN_SAMPLES else TEST_DIR
    if i < len(all_files):
        file_path = all_files[i]
    else:
        file_path = random.choices(all_files, k=1)[0]
    extract_clip(file_path, DURATION, SAMPLE_RATE, output_dir)

print("✅ 所有负样本生成完成！")
