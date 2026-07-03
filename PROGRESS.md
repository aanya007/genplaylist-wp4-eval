
# Progress Tracker — GenPlaylist WP-D Part 4 (Eval Matrix + Verbalization Pipeline)

## Goal
Build (1) a many-to-many evaluation matrix for generated vs ground-truth playlists
(CLAP-based OAS via Hungarian matching + FAD + CLAP-Sim), and (2) a single-shot
verbalization + synthesis pipeline that takes a predicted song embedding + creative
cues and produces lyrics + music attributes + generated audio via ACE-Step.

We are responsible for WP-D Part 4 within the larger GenPlaylist project:
  - Core diffusion model (backbone): mentor-led — we consume its GeneratedItem output
  - WP-A (input normalization): another student — we consume its ContextPrefix output
  - WP-B (creative cue mining): another student — we consume its item2cues.json output
  - WP-C (Gradio demo + user study): another student — our eval scores feed into their UI
  - WP-D Part 4 (eval matrix + verbalization + synthesis pipeline): US

## Repo Layout
- src/
  - schema.py             — data contracts (copied from mentor repo, read-only)
  - eval_matrix.py        — CLAP OAS + FAD + CLAP-Sim evaluation — COMPLETE
  - test_eval_matrix.py   — test for eval matrix
  - verbalization.py      — Qwen/DashScope single-shot verbalization — COMPLETE
  - test_verbalization.py — test for verbalization
  - synthesis.py          — ACE-Step synthesis wrapper — COMPLETE
  - test_synthesis.py     — full pipeline test (verbalization -> synthesis)
- data/curated/           — 25 curated input JSON files (gittracked)
- data/raw/               — raw audio + metadata (pod only, gitignored — too large)
- outputs/                — generated audio, scores (large files gitignored)
- reference/GenPlaylist   — mentor repo (read-only clone)
- reference/VibeMus       — ACE-Step agent repo (read-only clone)
- CLAUDE.md               — Claude Code standing instructions
- PROGRESS.md             — this file

## Mentor Repo Structure (tuteng0915/GenPlaylist)
Key files we read + integrated with:
- src/00_data_schema/schema.py         — COPIED to src/schema.py. Defines:
    CatalogItem: one catalog song (title, artist, genre, mood, tempo, key,
                 language, lyric_excerpt, audio_path, tags, feature_index)
    GeneratedItem: diffusion model output (rvq_codes, z_hat_emb dim=64,
                   mu_c_emb dim=64, sigma_c2, cue_ids, context_prefix)
    SynthesisResult: synthesis output (audio_path, music_attributes,
                     lyric_draft, neighbors, style_summary, generated_item)
    ContextPrefix: standardized playlist prefix (item_ids, source, raw_input)
    Constants: CLHE_EMB_DIM=64, RQ_N_CODEBOOKS=3, RQ_CODEBOOK_SIZE=256
- src/04_synthesis/verbalization.py    — COPIED to src/verbalization.py, adapted
- src/04_synthesis/synthesis.py        — COPIED to src/synthesis.py, adapted

---

## STEP 0 — Environment Setup — COMPLETE

### What was done
- Created GitHub repo (genplaylist-wp4-eval) with folder structure,
  CLAUDE.md (Claude Code project instructions), .gitignore, PROGRESS.md
- Resolved RunPod SSH access on a shared pod: could not overwrite existing
  SSH keys on the shared RunPod account. Fix: used web terminal to APPEND
  own SSH public key to pod's ~/.ssh/authorized_keys (additive, non-destructive)
- Installed Miniconda, created vibemus conda env (Python 3.10)
- Installed VibeMus dependencies in order:
    torch/torchvision/torchaudio (CUDA 11.8 build)
    pip install -r requirements.txt
    pip install transformers==4.51.0 (pinned last — ace-step wants 4.50.0 but
    4.51.0 works; pin last to override)
- Installed: ffmpeg (apt), hf_transfer (pip)
- Ran VibeMus via Gradio (share=True for headless pod), downloaded ACE-Step-v1-3.5B
  checkpoint (6.61GB transformer + 314MB DCAE + 206MB vocoder + 1.13GB UMT5)
- Smoke test PASSED: manually entered tags + lyrics, clicked Generate, produced
  a playable .wav file with no CUDA/ffmpeg errors
- Installed Claude Code on pod via npm (Node.js 20)
- Configured DASHSCOPE_API_KEY for Qwen/VibeMus LLM calls
- Configured ANTHROPIC_API_KEY for Claude Code agent use

### Mistakes + fixes encountered
- Miniconda URL wrong: must be repo.anaconda.com/miniconda/ NOT /miniconda3/
  (the /miniconda3/ path gives 404)
- conda ToS now mandatory before conda create works (not documented anywhere):
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
- hf_transfer not in VibeMus requirements.txt but required at runtime because
  ACE-Step sets HF_HUB_ENABLE_HF_TRANSFER=1 — install separately
- SSH sessions reset on laptop sleep: must re-run conda activate vibemus and
  cd to project root on every reconnect (env does not persist across sessions)
- POD WIPE: container was terminated (not just stopped), container ID changed
  from 9aff7bae037b to f3cab1d89a02, ALL files at / were lost. Only GitHub
  content survived. Fix: all work now lives in /workspace (RunPod persistent
  volume), always push code to GitHub before stopping pod
- Port forwarding not configured on RunPod pod — used Gradio share=True
  tunnel instead (gives a public gradio.live URL)
- SSH key on local Mac did not exist initially (no ~/.ssh/id_ed25519) —
  had to generate with ssh-keygen -t ed25519 first

---

## STEP 1 — Data Setup + Curated Dataset — COMPLETE (with caveats)

### Dataset overview
MPD-subset (from mentor's shared Google Drive):
- 6,585 playlists, 5,119 unique songs, avg playlist length 28.5 songs
- Audio: data/audio/spotify/{item_id}.mp3 (one mp3 per song)
- Lyrics: data/lyrics/spotify/{item_id}.txt (one txt per song)
- Playlists: data/playlists/mpd_subset/playlists.txt
  Format per line: playlist_id, song_id_1, song_id_2, ... (comma separated)
- Catalog: data/dataset/catalog_metadata.json (title/artist/genre/mood per song)
- Splits: data/dataset/splits/train.txt, val.txt, test.txt

### Data transfer process
- Downloaded MPD-subset split archive (audio_part_aa through audio_part_ak,
  ~4GB each) from shared Google Drive via rclone on pod:
    rclone copy "audio files:MPD-subset" /workspace/.../data/raw/ --drive-shared-with-me
  Note: --drive-shared-with-me flag required for folders shared TO you (not owned by you)
- Hit DISK QUOTA during extraction: only 1,147 of 5,119 mp3s extracted,
  all as 0-byte placeholder files (real content not written due to quota)
- Missing archive parts: audio_part_af through audio_part_aj never fully
  downloaded — these contain catalog_metadata.json and splits/
- Audio/lyrics ID mismatch in partial extract: the 1,147 extracted mp3 IDs
  don't match any of the 1,690 extracted lyrics IDs (zero overlap) — this is
  because we only got a partial archive, full dataset has matched IDs per README
- BLOCKER: need mentor to increase /workspace volume size on RunPod

### Curated dataset (25 JSON files) — built despite blocker
- Used Claude Code to parse playlists.txt, find playlists with audio coverage
- 388 of 6,585 playlists have at least 10 audio-available songs
- Selected 5 playlists:

| Playlist ID | Total Songs | Audio Available | Seed Songs (first 5)           | Ground Truth (next 5)          |
|-------------|-------------|-----------------|-------------------------------|-------------------------------|
| 6334        | 54          | 18              | 8751,11195,12304,21255,48545  | 79149,88362,94558,110564,123208|
| 3657        | 56          | 18              | 15253,31050,55099,86943,113046| 198921,213995,243183,4587,8800 |
| 8280        | 56          | 16              | 10473,11195,47529,95284,100358| 109591,119143,149361,165507,170734|
| 3122        | 51          | 15              | 9706,31737,41172,61274,74242  | 99136,110564,118516,147365,150154|
| 13867       | 55          | 15              | 8814,10464,16770,33255,98000  | 125522,137333,141267,148708,150009|

- 5 variants per playlist: same 5 cues with different weight profiles
  (each variant emphasises one cue at weight 1.0, others at 0.2-0.4)
- 25 JSON files in data/curated/p{playlist_id}_v{1-5}.json
- All 250 referenced mp3 paths confirmed to exist on disk at creation time

### Known caveats
- Creative cues are PLACEHOLDERS: generic words (energetic, melancholic, upbeat,
  dreamy, intense) — same across all playlists. Real cues should come from WP-B's
  item2cues.json output. Will be replaced once WP-B delivers.
- mp3 files are 0-byte (disk quota issue) — valid paths but no real audio content
- A few song IDs appear in multiple playlists (e.g. 11195 in both 6334 and 8280)
  — inherent to source data, not an error
- macOS AppleDouble sidecar files (._<id>.mp3) exist alongside real mp3s —
  must filter names starting with ._ in any script that counts/reads these files

---

## STEP 2 — LLM Backend — COMPLETE

Decision: kept DashScope/Qwen as the LLM backend (have working key, it's what
the mentor designed verbalization.py around). Claude API is used only for
Claude Code agent assistance, not in the pipeline itself.

DASHSCOPE_API_KEY set in reference/VibeMus/.env and exported as env var.
Verified: VibeMus chat flow works with real Qwen responses (not just stub output).

---

## STEP 3 — Verbalization Pipeline — COMPLETE

### What was built (src/verbalization.py)
Adapted from mentor's skeleton at reference/GenPlaylist/src/04_synthesis/verbalization.py.
The mentor had written the full structure with _call_qwen3() as a stub returning
"STUB OUTPUT". Our contribution: replaced the stub with a real DashScope/Qwen API call
and fixed the schema import path.

The pipeline takes a GeneratedItem (from the diffusion model) and returns
music attributes + lyrics for ACE-Step. Here is exactly what it does:

Step 1 — knn_verbalize(z_hat_emb, catalog_embs, catalog_metadata, k=5):
  - Takes the 64-dim predicted song embedding from the diffusion model
  - Computes cosine similarity against all 5,119 catalog song embeddings
  - Returns the 5 most similar catalog songs (CatalogItems)
  - These 5 songs describe WHERE in music space the generated song should land
  - e.g. if z_hat_emb is near indie pop songs, neighbors will be indie pop songs

Step 2 — knn_verbalize(mu_c_emb, catalog_embs, catalog_metadata, k=5):
  - Same process but using the PLAYLIST CENTROID embedding (mu_c_emb)
  - Returns 5 songs nearest to the playlist's average style
  - These describe the GLOBAL STYLE of the whole playlist
  - Used to keep the generated song coherent with the playlist, not just the
    predicted position

Step 3 — generate_music_attributes(neighbors, style_summary, sigma_c2):
  - Calls Qwen with: neighbor descriptions + style summary + dispersion hint
  - sigma_c2 (playlist dispersion): if high → playlist is diverse → give more
    creative latitude; if low → playlist is compact → stay close to style
  - Qwen returns: comma-separated ACE-Step style tags
  - e.g. "synth-pop, nostalgic, 100 BPM, retro synths and bassline, F minor, English"

Step 4 — generate_lyrics(neighbors, style_summary, music_attributes, sigma_c2):
  - Calls Qwen with: same context + the attributes from Step 3
  - Qwen returns: full ACE-Step markup lyrics with section markers
  - Format: [verse] / [chorus] / [bridge] on their own lines, each lyric line
    on its own line, blank line between sections

Step 5 — verbalize() wrapper combines all steps, returns:
  {neighbors, style_summary, music_attributes, lyric_draft}

### What we changed from mentor's skeleton
- Replaced _call_qwen3() stub with real DashScope SDK call:
    dashscope.Generation.call(model="qwen-plus", messages=[system+user])
- Fixed import path: sys.path.insert(0, os.path.dirname(__file__)) so
  schema.py is found correctly from src/ directory
- Indentation fix: mentor's commented-out code had extra indentation —
  carried over when uncommented, caused IndentationError on line 54

### Verified output (test_verbalization.py on mock data)
Music attributes: "synth-pop, nostalgic, 98 BPM, retro synths and bassline, F minor, English"
Lyric draft (excerpt):
  [verse]
  Neon ghosts in the rearview mirror
  Static hum on a broken speaker
  [chorus]
  We were analog in a digital age
  Flickering love on a dying stage
  [bridge]
  Time slows down—but never stops—just bends
  Like vinyl warping at the end
GeneratedItem validation: PASSED
Both Qwen API calls: SUCCEEDED

---

## STEP 4 — Evaluation Matrix — CODE COMPLETE (full-data run blocked)

### What was built (src/eval_matrix.py)
Main function: evaluate(generated_audio_paths, ground_truth_audio_paths,
                        prompt_text=None) -> {CLAP_OAS, FAD, CLAP_Sim}

### Metric 1: CLAP_OAS (Optimal Alignment Score)
This is the core many-to-many matching metric from the meeting notes requirement:
"find best mapping between 2 sets of sounds to maximise total/average similarity"

How it works:
1. Load CLAP model (laion_clap.CLAP_Module, enable_fusion=False,
   downloads ~630MB 630k-audioset checkpoint on first run)
2. Extract a 512-dim CLAP audio embedding for every generated song
   and every ground truth song
3. L2-normalize all embeddings
4. Build NxM cosine similarity matrix (N generated, M ground truth)
5. Run Hungarian algorithm (scipy.optimize.linear_sum_assignment on
   cost matrix = 1 - similarity) — finds optimal one-to-one matching
   that maximises total similarity across the two sets
6. Average the matched similarities = CLAP_OAS score (0 to 1)
Higher = generated songs are semantically closer to what users actually chose.
Directly implements DDBC Algorithm 3 (Appendix B of paper).

### Metric 2: FAD (Frechet Audio Distance)
How it works:
1. Load VGGish model via frechet_audio_distance library (auto-downloads)
2. Extract VGGish embeddings for all audio in both sets
3. Fit Gaussian (mean + covariance) to each set's embeddings
4. Compute Frechet distance between the two Gaussians
Lower FAD = generated audio distribution is closer to real music distribution.
Measures audio quality, not semantic matching.
Implementation note: frechet_audio_distance.score() takes DIRECTORIES not file
lists — we symlink each set into a temp dir with unique index-prefixed names
(to avoid cross-playlist collisions), score, then clean up the temp dir.

### Metric 3: CLAP_Sim (Condition Adherence)
How it works:
1. Use same CLAP model to get text embedding of prompt_text
   (text passed as 2-item list — single-item list trips batchnorm in model)
2. Compute cosine similarity between each generated song's audio embedding
   and the text embedding
3. Average across all generated songs = CLAP_Sim score
Higher = generated audio matches the text prompt used to generate it.
Only computed when prompt_text is provided.

### Why NOT ImageBind
ImageBind was originally planned (paper uses MERT + CLAP + ImageBind as
three converging similarity signals, meeting notes also mention ImageBind).
Dropped because imagebind package has a pkg_resources dependency that
breaks with newer setuptools (v82+, which removed pkg_resources from default
install). Patch attempts: (1) pip install setuptools — pkg_resources still
not found; (2) nano data.py to replace import pkg_resources with
import importlib.resources — fixed import but revealed further chain of
dependency issues. Time cost exceeded benefit for this stage.
TODO: retry ImageBind once other blockers resolved — would strengthen
eval matrix and better match the paper.

### Why NOT MERT
MERT (m-a-p/MERT-v1-95M, ~400MB) was dropped early to keep eval matrix lean.
CLAP already covers audio-semantic similarity well for our scale (5,119 songs).
Can be added later as a third encoder if needed.

### Verified results (on 2 real non-zero mp3 files)
evaluate([8751.mp3, 79149.mp3], [79149.mp3, 8751.mp3],
         prompt_text='energetic upbeat pop'):
  CLAP_OAS: 0.404
  FAD: ~5e-12 (effectively 0 — expected, sets are same songs permuted)
  CLAP_Sim: 0.091
FAD near 0 on identical-but-permuted sets = correct behaviour (sanity check passed).

### Current blocker
All 1,147 mp3s are 0-byte due to disk quota issue. Full batch run blocked.
Full results table (25 inputs x 3 metrics) cannot be produced until
real audio is available.

---

## STEP 5 — Synthesis Pipeline — COMPLETE

### What was built (src/synthesis.py)
Adapted from mentor's skeleton at reference/GenPlaylist/src/04_synthesis/synthesis.py
and VibeMus's pipeline.py + tools.py.

Function: synthesize(music_attributes, lyric_draft, audio_duration=30.0,
                     style_ref_audio_path=None, output_dir="outputs",
                     filename=None) -> str (absolute path to .wav)

How it works:
1. On module import: loads ACEStepPipeline as singleton (device_id=0,
   dtype="bfloat16", torch_compile=False)
   Downloads on first run: 6.61GB transformer, 314MB DCAE, 206MB vocoder,
   1.13GB UMT5, ~16.8MB tokenizer
2. If style_ref_audio_path provided: uses ACE-Step edit task with the
   nearest catalog neighbor as acoustic style reference
3. Otherwise: standard text-to-music generation:
   pipe(prompt=music_attributes, lyrics=lyric_draft, audio_duration=audio_duration)
4. Moves output to output_dir/filename.wav if filename specified

ACE-Step call signature (verified from pipeline_ace_step.py __call__):
  prompt = comma-separated style tags string
  lyrics = ACE-Step markup string ([verse]/[chorus]/[bridge])
  audio_duration = float in seconds (random 30-240s if <= 0)
  Returns list; first element is output path

### Full pipeline verified end-to-end (test_synthesis.py)
Mock data -> verbalization (2 Qwen API calls) -> ACE-Step synthesis:
  Music attributes: "synth-pop, nostalgic, 100 BPM, retro synths and bassline, F minor, English"
  ACE-Step: 60 diffusion steps at 11.96 it/s (~5 seconds of compute)
  Output: outputs/test/test_synthesis_output.wav
  File size: 11.0 MB (30 seconds of audio)
  GPU memory: 7.43GB allocated, 7.86GB reserved
  Total time: ~27 seconds wall clock

---

## STEP 6 — Full Batch Run — BLOCKED

Run all 25 curated inputs through the full pipeline:
verbalize() -> synthesize() -> evaluate()
Produce results table (25 rows x metrics).

Blocked on: real mp3s needed (disk quota issue).
Once disk is fixed:
1. Re-extract audio from archive parts
2. Load real catalog embeddings (clhe_weight.npy) for verbalization kNN
3. Run batch script across 25 inputs
4. Produce results table

---

## STEP 7 — Human User Study — BLOCKED

Waiting on WP-C demo system (another student).
Planned design:
- Pairwise comparison: our generated song vs retrieval baseline
- Rating forms: coherence, quality, overall satisfaction (5-point Likert)
- Evaluator demographics: listening habits, musical background
- Confirm between evaluator: given ground truth + generated, which fits playlist better
Will implement evaluation forms inside WP-C's Gradio app.

---

## Known Blockers (priority order)

1. CRITICAL — /workspace disk quota exceeded
   All mp3s are 0-byte. Batch run, real verbalization, eval all blocked.
   Fix: ask mentor to increase RunPod /workspace volume allocation.

2. HIGH — Missing archive parts (audio_part_af through audio_part_aj)
   These contain catalog_metadata.json and splits/ needed for real verbalization
   (currently using mock CatalogItems in tests).
   Fix: re-download once disk space available.

3. MEDIUM — Creative cues are placeholder
   Generic words across all playlists, not per-playlist cues.
   Fix: replace with WP-B item2cues.json when Jiatong (WP-B) delivers it.

4. LOW — ImageBind not implemented
   Weakens eval matrix vs paper (paper uses 3 encoders, we use 1+FAD).
   Fix: retry pkg_resources patch once disk issue resolved.

5. LOW — faiss not implemented for kNN (using numpy cosine)
   Fine for 5,119 songs. Only needed if scaling to full 254k catalog.

---

## What Can Be Reported Right Now

COMPLETED deliverables (all verified with real API/GPU calls):

1. Evaluation matrix (src/eval_matrix.py)
   - CLAP-based OAS with Hungarian optimal matching (DDBC Algorithm 3)
   - FAD (Frechet Audio Distance) for audio quality
   - CLAP-Sim for text prompt condition adherence
   - Verified: CLAP_OAS=0.404, FAD~0, CLAP_Sim=0.091 on 2 real audio files

2. Verbalization pipeline (src/verbalization.py)
   - Single-shot function: GeneratedItem -> music attributes + lyrics
   - kNN catalog lookup (cosine similarity, k=5 neighbors)
   - Two Qwen/DashScope LLM calls (attributes then lyrics)
   - ACE-Step markup output format verified correct
   - Integrated with mentor's schema.py data contracts
   - Verified: valid attributes + full verse/chorus/bridge lyrics produced

3. Synthesis pipeline (src/synthesis.py)
   - ACE-Step wrapper, adapted from VibeMus
   - Single function: (attributes, lyrics) -> .wav file path
   - Full chain verified: mock data -> 11MB .wav in ~27 seconds on GPU

PENDING (blocked on disk space):
- Full batch run (25 inputs -> results table)
- Real catalog embeddings for verbalization (needs catalog_metadata.json)
- Batch eval scores

Repo: github.com/aanya007/genplaylist-wp4-eval

---

## Full Log (chronological)

### 2026-06-27
- Created GitHub repo (genplaylist-wp4-eval)
- Added .gitignore, CLAUDE.md, PROGRESS.md, folder structure

### 2026-06-29
- Resolved RunPod SSH on shared pod (appended own key to authorized_keys)
- Installed Miniconda (fixed URL: /miniconda/ not /miniconda3/)
- Accepted conda ToS (mandatory, undocumented)
- Created vibemus conda env, installed all VibeMus deps
- Fixed hf_transfer missing dep
- Downloaded ACE-Step-v1-3.5B checkpoint
- Smoke test PASSED: Gradio UI, generated audio, no errors
- Installed Claude Code on pod (npm, Node.js 20)
- Configured DASHSCOPE_API_KEY

### 2026-07-03 (morning)
- Pod TERMINATED (container wiped, not just stopped)
- Container ID: 9aff7bae037b -> f3cab1d89a02
- GitHub only had 3 files (no src/, no data/)
- Recovery: re-cloned to /workspace, redid full environment setup
- All work now in /workspace (persistent volume)
- Installed rclone, configured Google Drive OAuth (headless mode)
- Transferred MPD-subset via rclone --drive-shared-with-me
- Hit disk quota: 1,147 mp3s extracted as 0-byte placeholders
- Deleted raw archive parts to free space (now have 112T free)
- Installed CLAP (laion_clap — loads ok)
- Attempted ImageBind: dropped (pkg_resources chain failure)
- Used Claude Code to parse playlists.txt, find 388 candidate playlists
- Discovered audio/lyrics ID disjoint issue (zero overlap in partial extract)
- Selected 5 playlists, built 25 curated JSON files in data/curated/
- Implemented src/eval_matrix.py (CLAP OAS + FAD + CLAP-Sim)
- Verified eval matrix on 2 real mp3s: CLAP_OAS=0.404, FAD~5e-12, CLAP_Sim=0.091
- Read mentor's schema.py + verbalization.py skeleton
- Copied both into src/, replaced _call_qwen3 stub with real DashScope call
- Fixed IndentationError in verbalization.py (extra indent from commented code)
- Ran test_verbalization.py: PASSED
  - "synth-pop, nostalgic, 98 BPM, retro synths and bassline, F minor, English"
  - Full verse/chorus/bridge lyrics generated correctly

### 2026-07-03 (afternoon)
- Read VibeMus tools.py + pipeline_ace_step.py __call__ signature to understand
  exact ACE-Step pipe() call parameters
- Implemented src/synthesis.py (ACE-Step wrapper, clean single function)
- Implemented src/test_synthesis.py (full verbalization -> synthesis chain)
- Ran test_synthesis.py: PASSED
  - ACEStepPipeline loaded successfully (weights re-downloaded ~8GB total)
  - 60 diffusion steps at 11.96 it/s
  - Output: outputs/test/test_synthesis_output.wav (11.0 MB, 30 seconds)
  - GPU: 7.43GB allocated, 7.86GB reserved
- Full pipeline end-to-end VERIFIED on mock data

### Next session priorities
1. Talk to mentor about /workspace disk quota increase (CRITICAL)
2. Talk to Jiatong (WP-B) about creative cue mining status
3. Once disk fixed: re-extract audio, load real catalog embeddings
4. Run batch: 25 inputs -> generated audio -> eval scores -> results table
5. Retry ImageBind installation
PROGRESSEOF
