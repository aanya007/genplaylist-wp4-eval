"""eval_matrix.py — Evaluation matrix for GenPlaylist WP-D Part 4.

Metrics:
  - CLAP_OAS   : CLAP-based Optimal Alignment Score (Hungarian matching)
  - ImageBind_OAS : ImageBind-based Optimal Alignment Score (Hungarian matching)
  - FAD        : Frechet Audio Distance (audio quality)
  - CLAP_Sim   : CLAP condition adherence (generated audio vs text prompt)

Usage:
    from eval_matrix import evaluate
    scores = evaluate(generated_paths, ground_truth_paths, prompt_text="pop song")
"""

import os
import sys
import tempfile
import shutil
import numpy as np
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# ImageBind path
# ---------------------------------------------------------------------------
sys.path.insert(0, '/home/wjzhang/tt_workspace/ImageBind')

# ---------------------------------------------------------------------------
# CLAP
# ---------------------------------------------------------------------------

def _load_clap():
    import laion_clap
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    model.eval()
    return model

def _clap_audio_embeddings(model, audio_paths):
    import torch
    embs = model.get_audio_embedding_from_filelist(audio_paths, use_tensor=False)
    embs = np.array(embs)
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
    return embs / norms

def _clap_text_embedding(model, text):
    emb = model.get_text_embedding([text, text], use_tensor=False)
    emb = np.array(emb)[0:1]
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    return emb / norms

# ---------------------------------------------------------------------------
# ImageBind
# ---------------------------------------------------------------------------

def _load_imagebind():
    import torch
    from imagebind.models import imagebind_model
    from imagebind.models.imagebind_model import ModalityType
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model, device

def _imagebind_audio_embeddings(model, device, audio_paths):
    import torch
    from imagebind import data as ib_data
    from imagebind.models.imagebind_model import ModalityType

    inputs = {
        ModalityType.AUDIO: ib_data.load_and_transform_audio_data(audio_paths, device)
    }
    with torch.no_grad():
        embs = model(inputs)[ModalityType.AUDIO]
    embs = embs.cpu().numpy()
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
    return embs / norms

# ---------------------------------------------------------------------------
# Hungarian OAS
# ---------------------------------------------------------------------------

def _hungarian_oas(gen_embs, gt_embs):
    """Optimal Alignment Score via Hungarian algorithm (DDBC Algorithm 3)."""
    sim_matrix = gen_embs @ gt_embs.T          # (N, M)
    cost_matrix = 1.0 - sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_sims = sim_matrix[row_ind, col_ind]
    return float(np.mean(matched_sims))

# ---------------------------------------------------------------------------
# FAD
# ---------------------------------------------------------------------------

def _compute_fad(generated_paths, ground_truth_paths):
    from frechet_audio_distance import FrechetAudioDistance
    fad = FrechetAudioDistance(model_name="vggish", sample_rate=16000, verbose=False)

    gen_dir = tempfile.mkdtemp()
    gt_dir = tempfile.mkdtemp()
    try:
        for i, p in enumerate(generated_paths):
            ext = os.path.splitext(p)[1]
            os.symlink(os.path.abspath(p), os.path.join(gen_dir, f"{i:04d}{ext}"))
        for i, p in enumerate(ground_truth_paths):
            ext = os.path.splitext(p)[1]
            os.symlink(os.path.abspath(p), os.path.join(gt_dir, f"{i:04d}{ext}"))
        score = fad.score(gen_dir, gt_dir)
    finally:
        shutil.rmtree(gen_dir, ignore_errors=True)
        shutil.rmtree(gt_dir, ignore_errors=True)
    return float(score)

# ---------------------------------------------------------------------------
# Main evaluate function
# ---------------------------------------------------------------------------

def evaluate(
    generated_audio_paths,
    ground_truth_audio_paths,
    prompt_text=None,
    clap_model=None,
    imagebind_model=None,
    imagebind_device=None,
):
    """Evaluate generated songs against ground truth.

    Parameters
    ----------
    generated_audio_paths    : list of paths to generated .mp3/.wav files
    ground_truth_audio_paths : list of paths to ground truth .mp3/.wav files
    prompt_text              : optional text prompt used to generate songs
    clap_model               : optional preloaded CLAP model (avoids reload)
    imagebind_model          : optional preloaded ImageBind model
    imagebind_device         : device string for ImageBind ('cuda' or 'cpu')

    Returns
    -------
    dict with keys: CLAP_OAS, ImageBind_OAS, FAD, CLAP_Sim
    """
    results = {}

    # --- CLAP embeddings ---
    print("[eval] Loading CLAP...")
    clap = clap_model or _load_clap()
    gen_clap = _clap_audio_embeddings(clap, generated_audio_paths)
    gt_clap = _clap_audio_embeddings(clap, ground_truth_audio_paths)
    results['CLAP_OAS'] = _hungarian_oas(gen_clap, gt_clap)

    # --- ImageBind embeddings ---
    print("[eval] Loading ImageBind...")
    if imagebind_model is None:
        ib_model, ib_device = _load_imagebind()
    else:
        ib_model, ib_device = imagebind_model, imagebind_device
    gen_ib = _imagebind_audio_embeddings(ib_model, ib_device, generated_audio_paths)
    gt_ib = _imagebind_audio_embeddings(ib_model, ib_device, ground_truth_audio_paths)
    results['ImageBind_OAS'] = _hungarian_oas(gen_ib, gt_ib)

    # --- FAD ---
    print("[eval] Computing FAD...")
    results['FAD'] = _compute_fad(generated_audio_paths, ground_truth_audio_paths)

    # --- CLAP-Sim ---
    if prompt_text:
        print("[eval] Computing CLAP-Sim...")
        text_emb = _clap_text_embedding(clap, prompt_text)
        sims = gen_clap @ text_emb.T
        results['CLAP_Sim'] = float(np.mean(sims))
    else:
        results['CLAP_Sim'] = None

    return results
