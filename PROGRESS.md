# Progress Tracker — GenPlaylist WP-D Part 4 (Eval Matrix + Verbalization Pipeline)

## Goal
Build (1) a many-to-many evaluation matrix for generated vs ground-truth playlists
(CLAP-based OAS via Hungarian matching + FAD + CLAP-Sim), and (2) a verbalization
pipeline that takes a predicted song embedding + creative cues and produces
lyrics + music attributes for ACE-Step synthesis.

We are responsible for WP-D Part 4 within the larger GenPlaylist project:
  - Core preference modeling + diffusion model: mentor-led (we consume its output)
  - WP-A (input normalization): another student (we consume its output)
  - WP-B (creative cue mining): another student (we consume its output)
  - WP-C (demo + user study): another student (our eval feeds into their UI)
  - WP-D Part 4 (eval matrix + verbalization pipeline): US

## Repo Layout
- src/                    — our pipeline code
  - eval_matrix.py        — CLAP OAS + FAD + CLAP-Sim evaluation (COMPLETE)
  - test_eval_matrix.py   — test script for eval matrix
  - verbalization.py      — Qwen/DashScope verbalization pipeline (IN PROGRESS)
  - test_verbalization.py — test script for verbalization
  - schema.py             — copied from mentor repo (data contracts)
- data/curated/           — 25 curated input JSON files (5 playlists x 5 variants)
- data/raw/               — raw audio + metadata (on pod only, gitignored)
- outputs/                — generated audio, scores (gitignored for large files)
- reference/GenPlaylist   — mentor repo (read-only)
- reference/VibeMus       — ACE-Step agent repo (read-only)
- CLAUDE.md               — Claude Code project instructions
- PROGRESS.md             — this file

## Mentor Repo Structure (tuteng0915/GenPlaylist)
Key files we need to integrate with:
- src/00_data_schema/schema.py         — data contracts (CatalogItem, GeneratedItem,
                                          SynthesisResult, ContextPrefix) — COPIED to src/
- src/03_backbone_recommender/         — diffusion model (mentor-led, we consume output)
- src/04_synthesis/verbalization.py    — our verbalization skeleton (COPIED + adapted)
- src/04_synthesis/synthesis.py        — ACE-Step synthesis (TODO)
- src/04_synthesis/app.py              — WP-C Gradio demo (another student)

---

## STEP 0 — Environment Setup — COMPLETE

### What was done
- Created GitHub repo (genplaylist-wp4-eval) with structure:
  src/, data/curated/, data/raw/, outputs/, reference/, notes/
- Added CLAUDE.md (Claude Code project instructions), .gitignore, PROGRESS.md
- Resolved RunPod SSH access on shared pod: couldn't overwrite existing SSH keys
  on shared account — used web terminal to APPEND own key to pod's
  ~/.ssh/authorized_keys instead (additive, non-destructive)
- Installed Miniconda on pod
- Created vibemus conda env (Python 3.10)
- Installed VibeMus dependencies: torch/torchvision/torchaudio (cu118),
  requirements.txt, transformers==4.51.0
- Installed ffmpeg, hf_transfer
- Ran VibeMus end-to-end via Gradio (share=True), downloaded ACE-Step-v1-3.5B
  checkpoint, generated first test audio with no errors
- Installed Claude Code on pod via npm (Node.js 20)
- Configured rclone for Google Drive (headless OAuth flow on pod)
- Configured DashScope API key in .env for VibeMus chat flow
- Installed Claude Code, set ANTHROPIC_API_KEY for Claude Code agent use

### Mistakes / fixes
- Miniconda URL was wrong in common instructions: must use
  repo.anaconda.com/miniconda/ NOT /miniconda3/ (404 error otherwise)
- conda ToS acceptance now required before conda create will work:
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
- hf_transfer not in VibeMus requirements.txt but required at runtime for
  ACE-Step checkpoint download (HF_HUB_ENABLE_HF_TRANSFER=1 is set by ace-step)
- SSH sessions time out when laptop sleeps — must re-run conda activate vibemus
  and cd to project root on every reconnect
- Pod was TERMINATED (not just stopped) at some point — container ID changed
  from 9aff7bae037b to f3cab1d89a02, all files at / were wiped. Recovered by
  re-cloning repo. Going forward: all work in /workspace (persistent volume),
  always push code to GitHub before stopping pod

---

## STEP 1 — Data Setup — PARTIALLY COMPLETE (blocked on disk space)

### What was done
- Transferred MPD-subset from shared Google Drive to pod via rclone
  (used --drive-shared-with-me flag since folder is shared TO us, not owned by us)
- Dataset: 6,585 playlists, 5,119 unique songs, avg playlist length 28.5
- Extracted genplaylist_data_meta.tar.gz: got lyrics/, playlists/ directories
- Extracted 1,147 mp3 files before hitting disk quota
- Deleted raw split parts to free space after partial extraction

### Mistakes / blocks
- DISK QUOTA: /workspace volume hit allocation limit during archive extraction.
  Mp3s extracted as 0-byte placeholder files (not real audio).
  Need mentor to increase /workspace volume size on RunPod.
  Current state: 1,147 mp3 file entries exist but are all 0 bytes.
- Missing archive parts: audio_part_af through audio_part_aj never fully
  downloaded (failed in original rclone transfer). These parts contain
  data/dataset/catalog_metadata.json and splits/ files needed for proper
  playlist/song metadata.
- Audio/lyrics ID mismatch: the 1,147 extracted mp3 IDs don't match the
  1,690 extracted lyrics file IDs — zero overlap. This is because we only
  got a partial archive. Full archive has matched IDs per the README.
- rclone initially configured wrong (used browser auth flow on headless pod
  which fails). Fix: must select N when asked "Use web browser to authenticate"
  then run rclone authorize "drive" "..." on local Mac to get token.

### What we have
- data/raw/data/lyrics/spotify/{id}.txt — 1,690 lyrics files
- data/raw/data/playlists/mpd_subset/playlists.txt — full playlist structure
  (format: playlist_id, song_id_1, song_id_2, ... comma separated)
- data/raw/data/playlists/mpd_subset/item_ids.txt — all song IDs
- data/raw/data/playlists/mpd_subset/item_freq.json — song frequency counts
- 1,147 mp3 files at data/raw/data/audio/spotify/{id}.mp3 (all 0 bytes)

### What still needs to happen
- Mentor to increase /workspace volume size
- Re-extract audio properly once disk space is available
- Download missing parts (audio_part_af through audio_part_aj) for catalog metadata

---

## STEP 2 — Curated Dataset (25 Input Files) — COMPLETE

### What was done
- Used Claude Code to parse playlists.txt and find playlists with audio coverage
- 388 of 6,585 playlists have at least 10 audio-available songs
- Selected 5 playlists (by audio count):

| Playlist ID | Total Songs | Audio Available |
|-------------|-------------|-----------------|
| 6334        | 54          | 18              |
| 3657        | 56          | 18              |
| 8280        | 56          | 16              |
| 3122        | 51          | 15              |
| 13867       | 55          | 15              |

- For each playlist: first 5 audio-available songs = seed, next 5 = ground truth
- 5 variants per playlist by varying creative cue weights
- 25 JSON files written to data/curated/ (all 250 referenced mp3 paths confirmed
  to exist on disk at time of creation — note: files are 0-byte due to disk issue)
- Schema: {playlist_id, variant_id, seed_songs:[{id, audio_path}],
           ground_truth_songs:[{id, audio_path}], creative_cues:[{cue, weight}]}

### Known limitations / caveats
- Creative cues are PLACEHOLDER (energetic, melancholic, upbeat, dreamy, intense)
  across all playlists — real per-playlist cues should come from WP-B output.
  Will be updated once WP-B delivers item2cues.json.
- A few song IDs appear in multiple playlists (e.g. 11195 seeds both 6334 and
  8280) — inherent to source data, not an error.
- Audio files are 0-byte until disk space is fixed.

---

## STEP 3 — Evaluation Matrix — CODE COMPLETE, BLOCKED ON AUDIO

### What was built (src/eval_matrix.py)

Main function: evaluate(generated_audio_paths, ground_truth_audio_paths,
                        prompt_text=None) -> {CLAP_OAS, FAD, CLAP_Sim}

Three metrics:

1. CLAP_OAS (Optimal Alignment Score)
   - Load CLAP (laion_clap.CLAP_Module, enable_fusion=False)
   - Extract audio embeddings for all generated + ground truth songs
   - Build cosine similarity matrix (generated x ground_truth)
   - Run Hungarian algorithm (scipy.optimize.linear_sum_assignment,
     cost = 1 - similarity) for optimal many-to-many matching
   - Average matched similarities = CLAP_OAS score
   - Higher = generated songs semantically closer to what users chose
   - Directly implements DDBC Algorithm 3 (Appendix B of paper)

2. FAD (Frechet Audio Distance)
   - Uses frechet_audio_distance library (VGGish model)
   - Measures audio quality: compares embedding distributions of
     generated vs real music sets
   - Lower FAD = better audio quality
   - Note: FAD.score() takes directories, not file lists — implementation
     symlinks each set into temp dirs, scores, then cleans up

3. CLAP_Sim (Condition Adherence)
   - Only computed when prompt_text is provided
   - Cosine similarity between CLAP audio embedding of each generated song
     and CLAP text embedding of the prompt used to generate it
   - Measures whether ACE-Step actually followed the text prompt
   - Higher = better prompt adherence

Also includes optional model-reuse path (pass preloaded models as args)
to avoid reloading CLAP + VGGish on every call during batch runs.

### Verification results (on 2 valid non-zero mp3 files)
CLAP_OAS: 0.404, FAD: ~5e-12, CLAP_Sim: 0.091
- FAD near 0 expected when comparing songs to themselves (sanity check passed)
- CLAP_OAS 0.404 reasonable for non-identical songs
- CLAP_Sim 0.091 reasonable for generic prompt on unrelated audio

### Why NOT ImageBind
ImageBind was originally planned as a third encoder (paper uses MERT + CLAP
+ ImageBind as three converging similarity signals). Dropped because:
- imagebind package has pkg_resources dependency that breaks with newer
  setuptools (setuptools 82+ removed pkg_resources from default install)
- Patching data.py fixed the import but revealed further dependency chain issues
- Time cost of debugging outweighed benefit for this stage
- CLAP + FAD gives a solid, defensible eval matrix for the deliverable
TODO: revisit ImageBind once disk issue is fixed and real audio is available.
Adding it would strengthen the eval matrix and match the paper more closely.

### Why NOT MERT
MERT (m-a-p/MERT-v1-95M) was dropped early — decision made to keep eval
matrix lean (CLAP already covers audio-semantic similarity well, MERT would
be redundant for our scale). Could be added later if needed.

### Current blocker
All 1,147 mp3s are 0-byte placeholder files due to disk quota issue.
Full batch run (all 25 curated inputs) blocked until real audio is available.
test_eval_matrix.py works correctly on the 2 files that had real content
before quota was hit.

---

## STEP 4 — Verbalization Pipeline — IN PROGRESS

### What was done
- Read mentor's verbalization.py skeleton
  (reference/GenPlaylist/src/04_synthesis/verbalization.py)
- Read schema.py (src/00_data_schema/schema.py) — defines:
    CatalogItem: metadata for one catalog song (title, artist, genre, mood,
                 tempo, key, language, lyric_excerpt, audio_path, tags)
    GeneratedItem: output of diffusion model (rvq_codes, z_hat_emb, mu_c_emb,
                   sigma_c2, cue_ids, context_prefix)
    SynthesisResult: output of synthesis step (audio_path, music_attributes,
                     lyric_draft, neighbors, style_summary)
    ContextPrefix: standardized playlist prefix (item_ids, source, raw_input)
- Copied both files into src/
- Replaced _call_qwen3() stub with real DashScope/Qwen API call

### What the verbalization pipeline does (step by step)
Given a GeneratedItem from the diffusion model:
1. knn_verbalize(z_hat_emb): find 5 nearest catalog songs to predicted embedding
   (these describe WHERE in music space the new song should land)
2. knn_verbalize(mu_c_emb): find 5 nearest catalog songs to playlist centroid
   (these describe the GLOBAL STYLE of the playlist)
3. generate_music_attributes(): call Qwen with neighbors + style_summary +
   sigma_c2 → comma-separated ACE-Step style tags
   (e.g. "indie pop, melancholic, 95 BPM, guitar, E minor, English")
4. generate_lyrics(): call Qwen with same context + attributes →
   ACE-Step markup lyrics ([verse]/[chorus]/[bridge] format)
5. Return: {neighbors, style_summary, music_attributes, lyric_draft}

### Current status
- src/verbalization.py: copied + _call_qwen3 stub replaced with real Qwen call
- src/test_verbalization.py: written with mock GeneratedItem + 5 mock CatalogItems
- BLOCKED: IndentationError in verbalization.py (indentation from mentor's
  commented-out code block carried over incorrectly — being fixed)

### TODO after fixing IndentationError
- Run test_verbalization.py end to end
- Verify Qwen produces valid music_attributes (comma-separated tags) and
  lyric_draft (proper ACE-Step markup with section markers)
- Connect to real catalog: load catalog_metadata.json + clhe_weight.npy
  (need missing archive parts for catalog_metadata.json)

---

## STEP 5 — Synthesis Pipeline — NOT STARTED

Connect verbalization output to ACE-Step to generate real audio.
Will adapt reference/VibeMus/pipeline.py into src/synthesis.py.
Needs GPU + working audio files.

Function signature (per mentor schema):
synthesize(music_attributes, lyric_draft, output_path) -> SynthesisResult

---

## STEP 6 — Full Pipeline End-to-End Run — NOT STARTED

For each of 25 curated inputs:
1. verbalize() → music_attributes + lyric_draft
2. synthesize() → generated .wav file
3. evaluate() → {CLAP_OAS, FAD, CLAP_Sim} scores
Produce results table. Sanity check: self-similarity test.
Blocked on: disk space fix + real audio + synthesis implementation.

---

## STEP 7 — Human User Study — BLOCKED

Waiting on WP-C demo system (another student).
Design: pairwise comparison (our generated vs retrieval baseline),
        rating forms (coherence, quality, overall satisfaction),
        evaluator demographics (listening habits, musical background).
Will implement evaluation forms inside WP-C's Gradio app.

---

## Known Issues / Blockers (priority order)

1. CRITICAL: /workspace disk quota exceeded — mp3s are 0-byte.
   Fix: ask mentor to increase RunPod /workspace volume size.
   Impact: blocks eval matrix batch run, synthesis, full pipeline.

2. HIGH: Missing archive parts (audio_part_af through audio_part_aj).
   Contains catalog_metadata.json needed for real verbalization.
   Fix: re-download from Drive once disk space is available.

3. MEDIUM: IndentationError in src/verbalization.py line 54.
   Fix: in progress (correct indentation of _call_qwen3 function body).

4. MEDIUM: Creative cues are placeholder.
   Fix: replace with WP-B output (item2cues.json) when available.
   Contact: Jiatong (WP-B owner) to check if cue mining is done.

5. LOW: ImageBind not implemented.
   Fix: retry pkg_resources patch once other blockers resolved.
   Impact: weakens eval matrix slightly (paper uses 3 encoders, we use 1+FAD).

6. LOW: faiss not implemented for kNN (using numpy cosine instead).
   Fine for 5,119 songs. Only needed if scaling to full 254k catalog.

---

## What Can Be Reported Right Now

- Environment fully set up on RunPod GPU pod (VibeMus + ACE-Step verified
  working end-to-end, generated real audio in smoke test)
- 25 curated test inputs built from real Spotify MPD playlist data
  (5 playlists x 5 creative cue weight variants, proper seed/GT split)
- Evaluation matrix implemented (src/eval_matrix.py):
    CLAP-based OAS with Hungarian optimal matching (per DDBC Algorithm 3)
    FAD (Frechet Audio Distance) for audio quality
    CLAP-Sim for condition adherence
    Verified working on valid audio (CLAP_OAS=0.404, FAD≈5e-12, CLAP_Sim=0.091)
- Verbalization pipeline adapted from mentor skeleton (src/verbalization.py):
    kNN catalog lookup (numpy cosine, k=5)
    Qwen/DashScope LLM calls for music attributes + ACE-Step lyrics
    Full integration with mentor's schema.py data contracts
- Schema.py integrated (CatalogItem, GeneratedItem, SynthesisResult,
  ContextPrefix understood and in use)
- Repo: github.com/aanya007/genplaylist-wp4-eval

## Log

### 2026-06-27
- Set up repo, CLAUDE.md, .gitignore, PROGRESS.md

### 2026-06-29
- Resolved RunPod SSH (appended key to authorized_keys via web terminal)
- Installed Miniconda (fixed URL), conda ToS, vibemus env, all VibeMus deps
- Fixed hf_transfer missing dep, downloaded ACE-Step checkpoint
- Smoke test passed: generated audio via Gradio UI
- Installed Claude Code via npm, configured ANTHROPIC_API_KEY
- Configured rclone for Google Drive (headless OAuth)
- Configured DASHSCOPE_API_KEY for VibeMus chat

### 2026-07-03
- Pod wiped (container terminated, not just stopped)
- Recovered: re-cloned into /workspace (persistent volume going forward)
- Reinstalled Miniconda, conda env, all deps
- Transferred MPD-subset via rclone --drive-shared-with-me
- Hit disk quota: mp3s extracted as 0-byte files
- Deleted raw archive parts to free space
- Installed CLAP (laion-clap confirmed loading)
- Attempted ImageBind: dropped due to pkg_resources issue
- Built curated dataset: 25 JSON files via Claude Code
- Implemented eval_matrix.py: CLAP OAS + FAD + CLAP_Sim
- Verified eval matrix on 2 valid mp3s (CLAP_OAS=0.404, FAD≈5e-12)
- Read mentor schema.py + verbalization.py skeleton
- Copied + adapted verbalization.py (replaced _call_qwen3 stub with Qwen call)
- Written test_verbalization.py with mock data
- Hit IndentationError in verbalization.py — fixing

### Next session
- Fix IndentationError in verbalization.py
- Run test_verbalization.py end to end (verify Qwen produces valid output)
- Talk to mentor about disk space increase
- Talk to Jiatong (WP-B) about cue mining status
- Once disk fixed: re-extract audio, run eval matrix batch, start synthesis
