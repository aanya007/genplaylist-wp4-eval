"""Test eval_matrix.py on real audio files from the dataset."""

import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from eval_matrix import evaluate

# Load a curated input
DATA_DIR = '/home/wjzhang/tt_workspace/data/data/audio/spotify'
CURATED = '/home/wjzhang/tt_workspace/genplaylist-wp4-eval/data/curated/p6334_v1.json'

with open(CURATED) as f:
    curated = json.load(f)

# Build real paths
def real_path(song_id):
    return os.path.join(DATA_DIR, f"{song_id}.mp3")

generated_paths = [real_path(s['id']) for s in curated['seed_songs']
                   if os.path.exists(real_path(s['id'])) and
                   os.path.getsize(real_path(s['id'])) > 0]

gt_paths = [real_path(s['id']) for s in curated['ground_truth_songs']
            if os.path.exists(real_path(s['id'])) and
            os.path.getsize(real_path(s['id'])) > 0]

print(f"Generated paths found: {len(generated_paths)}")
print(f"Ground truth paths found: {len(gt_paths)}")

if not generated_paths or not gt_paths:
    print("ERROR: No valid audio files found. Check paths.")
    sys.exit(1)

print("\nRunning evaluation...")
scores = evaluate(
    generated_audio_paths=generated_paths,
    ground_truth_audio_paths=gt_paths,
    prompt_text='energetic upbeat pop'
)

print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)
for k, v in scores.items():
    print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: None")
