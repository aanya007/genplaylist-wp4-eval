##Progress Tracker — GenPlaylist WP-D Part 4 (Eval Matrix + Verbalization Pipeline)

## Goal
Build (1) a many-to-many evaluation matrix for generated vs ground-truth playlists
(MERT/CLAP/ImageBind similarity + DDBC-style optimal-matching/OAS), and (2) an adapted
VibeMus-based prompt/lyric generation pipeline, tested using real playlist audio with
manually-curated creative cues (standing in for WP-D's core diffusion model output,
which we assume to predict correctly per project scope).

## Repo Layout
- reference/GenPlaylist  — paper's reference repo (read-only)
- reference/VibeMus      — ACE-Step agent repo, adapted into our pipeline (read-only source)
- src/                   — our adapted pipeline code (single-shot verbalization, eval matrix)
- data/curated/          — curated dataset: real playlists/audio + manually-picked creative cues
- outputs/               — generated audio, lyrics, scores
- notes/                 — meeting notes, design notes

## Status: Step 0 — Environment Setup — COMPLETE

- [x] Set up project repo (genplaylist-wp4-eval), .gitignore, CLAUDE.md, PROGRESS.md
- [x] Clone GenPlaylist + VibeMus as reference
- [x] Get RunPod GPU pod access (shared pod; added own SSH key to authorized_keys
      via web terminal rather than overwriting existing account-level keys)
- [x] Install Miniconda on pod (note: correct URL is repo.anaconda.com/miniconda/...,
      not /miniconda3/...)
- [x] Accept conda Terms of Service for default channels (now required by newer conda
      before `conda create` will work)
- [x] Create `vibemus` conda env (python 3.10)
- [x] Install VibeMus dependencies: torch/torchvision/torchaudio (cu118), requirements.txt,
      transformers==4.51.0 (pinned last per README; conflicts with ace-step's stated
      4.50.0 requirement but works in practice)
- [x] Install ffmpeg
- [x] Install hf_transfer (undocumented missing dependency — needed because
      HF_HUB_ENABLE_HF_TRANSFER=1 is set for fast checkpoint downloads)
- [x] Run VibeMus (`python main.py`), expose Gradio UI via `share=True` (no direct
      SSH port forwarding set up; public gradio.live link used instead)
- [x] ACE-Step-v1-3.5B checkpoint downloaded successfully on first run
- [x] Smoke test passed: manually entered tags + lyrics in Gradio UI (no DASHSCOPE
      API key set), clicked Generate, produced a playable .wav with no CUDA/ffmpeg
      errors. Output saved to outputs/smoke_test/.

## Known deviations from VibeMus README (for reporting)
- Miniconda install URL in common instructions is outdated/wrong path.
- conda ToS acceptance step not mentioned anywhere, now mandatory.
- hf_transfer not listed in requirements.txt but required at runtime.
- No port-forwarding configured on RunPod; used Gradio share=True tunnel instead.
- Tested without DASHSCOPE_API_KEY (manual tag/lyric entry) since LLM backend
  swap to Claude API is the next step, not yet done.

## Status: Step 1 — Curated Dataset — NOT STARTED
- [ ] Locate/extract real audio + metadata files (audio_part_aa..ak, genplaylist_data_meta)
- [ ] Inspect metadata schema (playlist IDs, song IDs, audio file mapping)
- [ ] Pick 5 real playlists
- [ ] Split each into seed set (first 5 songs) / ground truth (next 5 songs)
- [ ] Manually curate 6-10 creative cues per playlist, with weighted variants (5 per playlist)
- [ ] Assemble 25 curated input JSON files in data/curated/

## Status: Step 2 — LLM Backend Swap (DashScope -> Claude API) — NOT STARTED
- [ ] Identify all DashScope/Qwen call sites in VibeMus (assistant.py, configs)
- [ ] Copy relevant files into src/, replace LLM calls with Anthropic API
- [ ] Set ANTHROPIC_API_KEY on pod
- [ ] Re-test chat flow end-to-end with Claude as backend

## Status: Step 3 — Pipeline Simplification (single-shot verbalization) — NOT STARTED
- [ ] Strip VibeMus's multi-turn chat loop into a single function:
      (seed_songs_metadata, weighted_creative_cues) -> (tags/attributes, lyrics)
- [ ] Wire ACE-Step synthesis to this single-shot function
- [ ] Run all 25 curated inputs through pipeline -> generated audio + lyrics + tags

## Status: Step 4 — Evaluation Matrix — NOT STARTED
- [ ] Set up MERT, CLAP, ImageBind encoders on pod
- [ ] Implement FAD (Frechet Audio Distance)
- [ ] Implement pairwise similarity matrix (generated set x ground-truth set)
- [ ] Implement Hungarian-algorithm optimal matching / OAS score (per DDBC Algorithm 3)
- [ ] Implement CLAP-Sim condition adherence check
- [ ] Wrap as reusable src/eval_matrix.py module

## Status: Step 5 — Run Full Mock Pipeline + Sanity Check — NOT STARTED
- [ ] Run 25 curated inputs through pipeline + eval matrix
- [ ] Produce results table, spot-check outputs by ear

## Status: Step 6 — Gradio Demo Frontend — NOT STARTED (low priority)

## Status: Step 7 — Human User Study — BLOCKED (waiting on WP-C demo system)

## Log

### 2026-06-29
- Set up project repo structure, .gitignore, CLAUDE.md.
- Resolved RunPod SSH access on shared pod.
- Completed full VibeMus environment setup on RunPod (see Step 0 above for details).
- Successfully generated first test audio via Gradio UI, no errors.
- Step 0 fully complete. Next: build curated dataset (Step 1), then swap LLM
  backend to Claude API (Step 2).

