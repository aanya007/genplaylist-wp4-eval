# GenPlaylist — Progress Tracker

## Goal

Build a reference-based personalized music generation pipeline. Given a set of seed songs, the system infers the user's musical taste (centroid μ_C and dispersion σ²_C), generates new song representations via dispersion-conditioned masked diffusion, and synthesizes playable audio via Qwen3 verbalization + ACE-Step.

**Dataset**: Spotify MPD v2 subset — 6,585 playlists / 5,119 songs  
**Stack**: CLHE embeddings → RVQ tokenizer (stride 11) → DIT diffusion → Qwen3 → ACE-Step

---

## Steps

### WP-A · Input Normalization (`01_input_normalization/`)
- [ ] `normalizer.py`: raw user input → `ContextPrefix(item_ids=[...])`
- [ ] Validate context prefix examples against `shared/schema.py`

### WP-B · Creative Cue Mining (`02_creative_cues/`)
- [ ] Scrape lyrics (Genius API for English, NetEase for Chinese; fallback to metadata)
- [ ] Extract raw cues via Qwen3-7B prompt (8–10 imagery words per track)
- [ ] Filter to 2048-word vocab: normalize → df filter (5 ≤ df ≤ 0.3N) → semantic dedup (cosine > 0.92) → top-2048 by IDF
- [ ] Assign K=6 cues per item (PMI sort + greedy diversity) → `item2cues.json`
- [ ] Output `cue_vocab.json` (2048 entries) and `item2cues.json`

### WP-D · Backbone Recommender (`03_backbone_recommender/`)

#### `tokenizer.py`
- [ ] Set vocab to 2894: BOS(1) + RVQ(768) + z_conf(74) + BOI(1) + EOS(1) + cues(2048) + MASK(1)
- [ ] Implement 11-token item stride: `[BOI, z1, z2, z3, z_conf, c1, c2, c3, c4, c5, c6]`
- [ ] Load `item2cues.json`; map cue IDs to vocab offset; fallback zeros when WP-B not ready
- [ ] `make_type_mask(seq_len, stride=11)` — legal token range per position type

#### `playlist_structure.py`
- [ ] `compute_playlist_structure()`: μ_C and σ²_C from CLHE embeddings per batch
- [ ] Wire into `tokenizer.tokenize_function()` collation (attach `mu_c`, `sigma_c2` to batch)
- [ ] Compute Q33/Q66 over training set → `dispersion_tiers.json`
- [ ] `get_dispersion_tier()` for compact / medium / diverse stratification at eval time

#### `models/dit.py`
- [ ] Add `DispersionEmbedder`: `W_mu: Linear(d_emb, d_hidden)` + `W_sigma: Linear(1, d_hidden)`, both SiLU
- [ ] Set `cond_dim = d_hidden`
- [ ] `forward`: fuse `c = SiLU(W_τ·τ) + SiLU(W_μ·μ_C) + SiLU(W_{σ²}·σ²_C)`

#### `diffusion.py`
- [ ] Forward corruption: exclude seed token positions from absorbing mask
- [ ] Reverse denoising: skip seed positions in `semi_ar_sample` / `_denoiser_step` unmask updates
- [ ] Pass `mu_c [B,d]` and `sigma_c2 [B,1]` through full call chain into `backbone.forward()`
- [ ] Apply `make_type_mask()` at inference to filter illegal logits

#### `dataset.py` + `configs/`
- [ ] Split each playlist: first ⌊|p|/2⌋ songs as reference context, first held-out next item as target
- [ ] Update `configs/config.yaml`: `vocab_size=2894`, `rq_n_codebooks=3`, `rq_codebook_size=256`, `stride=11`, `dispersion_cond=true`, `centroid_dim=512`
- [ ] Add ablation flags: `no_dispersion`, `no_cues`, `no_verbalization`, `no_diffusion`

### WP-C · Synthesis (`04_synthesis/`)

#### `verbalization.py`
- [ ] Wire real DashScope/Qwen3 call: `dashscope.Generation.call(model="qwen3-7b-instruct", ...)`
- [ ] Replace numpy cosine with faiss `IndexFlatIP`; build index once at startup
- [ ] Load `catalog_metadata.json` (title/artist/genre/mood/tempo/key/language/lyric_excerpt)
- [ ] Calibrate diversity threshold `sigma_c2 > Q66` (≈ 0.310 from training set)
- [ ] Per candidate: `verbalize()` independently; share one `style_summary` from μ_C kNN

#### `synthesis.py`
- [ ] Confirm ACE-Step install and test import
- [ ] Enable `torch_compile=True` (requires torch ≥ 2.3)
- [ ] Implement `style_ref_audio_path` support → ACE-Step `task='edit'` path
- [ ] Synthesize S candidates in parallel
- [ ] Wire `output_dir` from `configs/config.yaml`

#### `app.py`
- [ ] Gradio demo: seed song input → generated audio playback
- [ ] User study UI (5-point Likert: coherence / music quality / overall satisfaction)

### Evaluation (`evaluator.py`)
- [ ] Integrate MERT (`m-a-p/MERT-v1-95M`) as acoustic encoder
- [ ] Integrate CLAP (audio-language encoder)
- [ ] Integrate ImageBind (cross-modal encoder)
- [ ] `compute_semantic_sim(gen_audio, gt_audio, encoder)` → cosine similarity + GT upper bound
- [ ] **Dispersion Match Δσ²**: generated audio → CLHE → `|σ²_{C∪{ẑ}} − σ²_C|`
- [ ] **Centroid Distance CD**: for ablation table
- [ ] **FAD**: Fréchet Audio Distance vs. held-out real music; build genre reference set
- [ ] **Condition CLAP**: `cos(CLAP(audio), CLAP(attributes_text + lyrics))`
- [ ] Stratify Δσ² by compact / medium / diverse tiers → Table 3

### Pipeline (`pipeline/genplaylist.py`)
- [ ] End-to-end coordinator: `ContextPrefix → list[SynthesisResult]`
- [ ] Run `mock_run.py` to verify all WP interfaces connect cleanly
- [ ] Full end-to-end smoke test with 3 generated samples

---

## Log

### 2026-06-27
- Project scaffold committed (`src/`, `data/mock/`, `notes/`, `outputs/`, `reference/`)
- Reference implementations loaded: `reference/GenPlaylist/` (full source + docs) and `reference/VibeMus/` (ACE-Step pipeline)
- Design locked: vocab=2894, stride=11, L=3/K=256 RVQ, 2048-cue vocab, K=6 cues/item, dispersion via additive SiLU projection
- Dataset confirmed: MPD v2 subset — 6,585 playlists / 5,119 songs / avg σ²_C=0.282 (Q33=0.255, Q66=0.310)
