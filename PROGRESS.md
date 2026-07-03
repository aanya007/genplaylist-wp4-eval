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

## Status: Step 1 — Curated Dataset — COMPLETE (placeholder cues)
- [x] Locate/extract real audio + metadata files (audio_part_aa..ak, genplaylist_data_meta)
- [x] Inspect metadata schema (playlist IDs, song IDs, audio file mapping)
- [x] Pick 5 real playlists (6334, 3657, 8280, 3122, 13867)
- [x] Split each into seed set (first 5 audio-available songs) / ground truth (next 5 audio-available)
- [x] Curate creative cues per playlist, with weighted variants (5 per playlist)
      — NOTE: currently GENERIC PLACEHOLDER cues (energetic/melancholic/upbeat/dreamy/intense),
      not hand-curated per-playlist cues. Revisit once lyric matching is resolved.
- [x] Assemble 25 curated input JSON files in data/curated/

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


### 2026-07-03
- Pod was terminated/replaced (container ID changed from 9aff7bae037b to f3cab1d89a02).
- All files at / were wiped. GitHub only had scaffold files (.gitignore, CLAUDE.md, PROGRESS.md).
- Recovered by re-cloning repo into /workspace (persistent volume) and redoing environment setup.
- Going forward: all work in /workspace/genplaylist-wp4-eval to ensure persistence.

### 2026-07-03
- Pod was terminated/replaced (container ID changed from 9aff7bae037b to f3cab1d89a02).
- All files at / were wiped. GitHub only had scaffold files (.gitignore, CLAUDE.md, PROGRESS.md).
- Recovered by re-cloning repo into /workspace (persistent volume) and redoing environment setup.
- Going forward: all work in /workspace/genplaylist-wp4-eval to ensure persistence.

### 2026-07-03 — Playlist candidate scan (Step 1, read-only report)
- Parsed data/raw/data/playlists/mpd_subset/playlists.txt (6,585 playlists;
  format = `playlist_id, song_id, song_id, ...`).
- Cross-referenced against 1,147 mp3 files in data/raw/data/audio/spotify/.
- Found 388 playlists with >=10 audio-available songs; reported top 10 by audio count
  (best candidates: 6334/6388/3657 with 18 audio songs each).
- GOTCHA: both audio/ and lyrics/ dirs contain macOS AppleDouble sidecar files
  (`._<id>.mp3` / `._<id>.txt`) — inflates naive `ls` counts and must be filtered
  (skip names starting with `._`). Real counts: 1,147 audio, 1,690 lyrics.
- BLOCKER for lyric verbalization: audio song-IDs and lyric song-IDs are DISJOINT.
  Overlap = 0 across all 1,147 audio IDs (ranges overlap: audio 1-254002,
  lyrics 15-254042, but no common ID). => none of the audio-available playlist
  songs have matching lyrics. Lyric-based pipeline path can't use these audio songs
  as-is; need to source lyrics for the audio IDs (or audio for the lyric IDs), or
  confirm this is expected. Raised for decision before picking the final 5 playlists.
- No files modified (data untouched); report only.

### 2026-07-03 — Built 25 curated input JSONs (Step 1 complete)
- Selected 5 playlists: 6334, 3657, 8280, 3122, 13867.
- Per playlist: seed_songs = first 5 audio-available songs, ground_truth_songs = next 5
  audio-available songs (skipping playlist songs with no mp3). All 5 had >=10 audio songs.
- Wrote 25 files data/curated/p{playlist_id}_v{1-5}.json following schema
  {playlist_id, variant_id, seed_songs:[{id,audio_path}], ground_truth_songs:[{id,audio_path}],
  creative_cues:[{cue,weight}]}. audio_path = data/raw/data/audio/spotify/{id}.mp3.
- Validated: 25/25 JSON parse OK, all 250 referenced mp3 paths exist on disk.
- Creative cues are PLACEHOLDERS: 5 generic cues (energetic/melancholic/upbeat/dreamy/intense),
  same set in every file, with a different weight profile per variant (each variant emphasizes
  one cue at weight 1.0). To be replaced with real per-playlist creative cues later; lyrics
  still unmatched (see disjoint audio/lyric ID blocker above).
