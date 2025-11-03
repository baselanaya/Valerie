# Valerie Codebase Cleanup Analysis
## Comprehensive Audit of Obsolete, Redundant, and Outdated Files

**Date:** November 3, 2025  
**Analysis Scope:** Very thorough - all source files examined  
**Status:** System switched from VIDEO ASR to AUDIO-ONLY with Ensemble Distillation

---

## SECTION 1: VIDEO PROCESSING FILES (OBSOLETE)

All video-related modules are now obsolete as the system has been converted to audio-only.

### 1.1 SPATIO-TEMPORAL 3D CNN MODULE

**File:** `/home/user/Valerie/src/models/spatio_temporal.py` (418 lines)

**What it does:**
- Implements 3D CNN blocks (Conv3dBlock) for processing video frames [B, C, T, H, W]
- TemporalPositionalEncoding for temporal sequences
- SpatioTemporalEmbedding class combining 3D conv layers with positional encoding
- Designed to extract lip movement features from video input

**Why it's obsolete:**
- System now uses audio-only input (Mel spectrograms)
- 3D CNN is replaced by AudioFrontend (Mel spectrogram extraction)
- No longer needed for Stage 1 (Audio → Phonemes)

**What replaced it:**
- `src/models/audio_frontend.py` - MelSpectrogramExtractor for audio feature extraction
- `src/models/audio_phoneme_model.py` - AudioPhonemeASR without video processing

**Dependency chain:**
- Used by: `src/models/valerie_model.py` (still imports it)
- Also tested by: Multiple test files (see Section 4)

---

### 1.2 VALERIE MODEL (VIDEO-BASED)

**File:** `/home/user/Valerie/src/models/valerie_model.py` (360 lines)

**What it does:**
- Main integrated model for "Valerie Visual ASR"
- Integrates: SpatioTemporalEmbedding → Conformer → HybridCTCAttention
- Includes audio distillation from single Whisper teacher
- Expects video input: `video: torch.Tensor` [B, C, T, H, W]

**Why it's redundant:**
- Replaced by `AudioPhonemeASR` which uses audio instead of video
- Still imported and used by old training pipelines and tests
- Still exported in `src/models/__init__.py`

**What replaced it:**
- `src/models/audio_phoneme_model.py` (AudioPhonemeASR) - audio-based, 50M parameters
- More efficient and production-ready

**Where still used:**
1. `tests/test_mandatory_components.py` - Tests ValerieModel with video input
2. `tests/test_integration.py` - Integration tests with video
3. `train_multi_gpu.py` - Old multi-GPU training (root level)
4. `scripts/train_two_stage.py` - Old two-stage training script
5. `src/inference/inference_engine.py` - Inference uses ValerieModel
6. `src/visualization/model_analysis.py` - Model analysis tool
7. `examples/multi_gpu_training_comparison.py` - Old comparison example

---

### 1.3 VIDEO DATASET LOADERS

**File:** `/home/user/Valerie/src/data/dataset.py` (1,317 lines - LARGEST DATA FILE)

**What it does:**
- `EnhancedVoxCeleb2Dataset` - Loads VoxCeleb2 video files with transcriptions
- `VoxCeleb2Dataset` - Original VoxCeleb2 dataset loader
- `AVSpeechDataset` - Loads AVSpeech dataset (YouTube videos with audio)
- All use video frames, face detection, lip region extraction
- All return `DataSample` with `video_frames: torch.Tensor`

**Why it's obsolete:**
- System now uses LibriSpeech (audio-only, no video)
- Video loading functions no longer needed:
  - `_load_video_frames()` - OpenCV video processing
  - `_extract_face_region()` - Face detection for lip reading
  - `_load_video_frames_enhanced()` - Enhanced video loading
  
**Methods that are obsolete:**
```python
# Line 307: _load_video_frames_enhanced() - Video loading
# Line 352: _process_face_frame() - Face detection  
# Line 697: _load_video_frames() - Original video loading
# Line 772: extract_lip_region() - Lip extraction
# Line 1035: _extract_face_region() - AVSpeech face extraction
# Line 945: _download_video() - YouTube download (AVSpeech)
# Line 970: _extract_video_audio() - Video-to-audio extraction
```

**What replaced it:**
- LibriSpeech dataset loading (via Hugging Face datasets library)
- Used in: `scripts/train_stage1_phoneme_asr.py`

**Impact:**
- Entire file could be refactored to audio-only dataset loaders
- Currently maintains large video processing code that's not used

---

### 1.4 VIDEO TRANSFORMS

**File:** `/home/user/Valerie/src/data/transforms.py` (547 lines)

**What it does:**
- `VideoTransforms` class (lines 24-189):
  - Temporal cropping
  - Frame resizing and normalization
  - Mixup augmentation between videos
  - Horizontal flip, brightness/contrast adjustment
  - Designed for lip reading

**Why it's obsolete:**
- No longer needed for audio-only system
- Replaced by audio augmentation (SpecAugment, speed perturbation)

**Audio transforms (still needed):**
- `AudioTransforms` class (lines 191-332) - KEEP
- `DataAugmentationPipeline` (lines 397-505) - Can work with audio-only

**What to do:**
- Remove `VideoTransforms` class and its dependencies
- Keep `AudioTransforms`
- Update collate functions to remove video-specific handling

---

## SECTION 2: CONFLICTING CONFIGURATION FILES

### 2.1 VIDEO-BASED CONFIGS (OBSOLETE)

**File:** `/home/user/Valerie/configs/base_config.yaml`

**Conflicts:**
- References 3D CNN: `conv3d_channels: [64, 128, 256, 512]`
- References 3D kernel: `conv3d_kernel_size: 3`
- Video parameters: `input_height: 224`, `input_width: 224`
- Video preprocessing: `frame_rate: 25`
- Video augmentation: `horizontal_flip_prob: 0.5`, `brightness_range: 0.2`
- Face detection: `min_face_confidence: 0.8`
- Two-stage training ref: `use_two_stage_training: true` (for video→phonemes)
- Teacher: Single Whisper teacher (outdated)

**Status:** DEPRECATED - Should not be used for new training

---

**File:** `/home/user/Valerie/configs/avspeech_config.yaml`

**Conflicts:**
- AVSpeech dataset path (video dataset)
- Audio-visual synchronization: `sync_verification: true`, `max_sync_offset: 0.1`
- Video-specific: `use_face_coordinates: true`
- Face detection: `min_face_confidence: 0.9`
- Intended for: Video → Phonemes stage 1

**Status:** DEPRECATED - AVSpeech not used in audio-only system

---

**File:** `/home/user/Valerie/configs/voxceleb2_config.yaml`

**Conflicts:**
- VoxCeleb2 dataset (video dataset)
- Video preprocessing: `max_epochs: 120` (tuned for VoxCeleb2)
- Cross-modal attention: `cross_modal_attention: true`
- Video-specific: `alignment_loss_weight: 0.05`
- Teachers: References Wav2Vec2 as single teacher

**Status:** DEPRECATED - VoxCeleb2 not used in audio-only system

---

**File:** `/home/user/Valerie/configs/multi_gpu_config.yaml` (250 lines)

**Conflicts:**
- References base_config.yaml (video-based)
- Video preprocessing: `video_size: [224, 224]`
- Video parameters: `sequence_length: 150`
- Data augmentation: `use_augmentation: true` (implies video augmentation)
- Project name: `project_name: "valerie-visual-asr"` (outdated)

**Status:** PARTIALLY DEPRECATED - Can be adapted for audio but references wrong base config

---

### 2.2 CORRECT CONFIG (NEW STANDARD)

**File:** `/home/user/Valerie/configs/ensemble_distillation_config.yaml` (159 lines)

**What it provides:**
- ✅ Audio-only system
- ✅ Mel spectrogram parameters (n_fft, hop_length, n_mels)
- ✅ LibriSpeech dataset (audio-only)
- ✅ Ensemble distillation from 3 teachers:
  - Whisper Large V3 (weight: 0.4)
  - Wav2Vec2 Large (weight: 0.3)
  - HuBERT Large (weight: 0.3)
- ✅ Reduced student model (256 dim, 12 layers)
- ✅ Proper stage 1, 2, 3 training parameters
- ✅ SpecAugment configuration

**Status:** CORRECT - Should be the template for new configs

---

## SECTION 3: REDUNDANT AND DUPLICATE MODULES

### 3.1 DUPLICATE DISTILLATION MODULES

**Problem:** Two different distillation implementations exist

#### OLD: `/home/user/Valerie/src/models/audio_distillation.py` (557 lines)

**What it does:**
- `WhisperTeacher` - Single Whisper teacher for distillation
- `CrossModalAlignment` - CCA-based alignment for visual-audio feature alignment
- `AudioKnowledgeDistillation` - Single-teacher distillation
- Designed for: Visual model learning from audio (video model using audio teacher)

**Status:** OBSOLETE - Designed for VIDEO model distillation, not audio-only

**Used by:**
- `src/models/valerie_model.py` - Old video-based model
- Exported in `src/models/__init__.py`

---

#### NEW: `/home/user/Valerie/src/models/ensemble_distillation.py` (696 lines)

**What it does:**
- `TeacherModel` - Wrapper for any teacher (Whisper, Wav2Vec2, HuBERT, Auto)
- `EnsembleDistillationModule` - Multiple teachers with weighted aggregation
- Supports 3 or more teacher models simultaneously
- Temperature-scaled softmax for soft labels
- Feature matching loss (MSE between teacher and student)
- KL divergence for soft label distillation

**Status:** CORRECT - New ensemble approach for Stage 1 audio→phonemes

**Used by:**
- `src/models/audio_phoneme_model.py` - New audio model
- `scripts/train_stage1_phoneme_asr.py` - New training pipeline

---

**Recommendation:**
- Keep `ensemble_distillation.py` (more general, supports multiple teachers)
- Remove `audio_distillation.py` (legacy, single-teacher only)
- Update imports in `__init__.py`

---

## SECTION 4: OBSOLETE TEST FILES

Tests still testing removed video functionality (7 test functions need updating/removal):

### 4.1 `/home/user/Valerie/tests/test_mandatory_components.py`

**Obsolete tests:**
- Tests `ValerieModel` with video input
- Generates synthetic video: `video = torch.randn(batch_size, 3, 8, 224, 224)`
- Tests model forward pass with video
- Expects spatio-temporal features

**Action:** Update to test `AudioPhonemeASR` with audio instead

---

### 4.2 `/home/user/Valerie/tests/test_integration.py`

**Obsolete sections:**
- `generate_video_data()` (line 47-54)
- Video file handling (lines 88-99, 151-209)
- `test_video_to_phonemes_pipeline()` (lines 280-295)
- Video batch collation (lines 556-565)
- Full integration with ValerieModel and video

**Status:** 30-40% of this file tests obsolete video functionality

---

### 4.3 `/home/user/Valerie/tests/test_models.py`

**Obsolete test classes/methods:**
```
TestSpatioTemporalEmbedding:
  - test_conv3d_block()
  - test_temporal_positional_encoding()
  - test_spatio_temporal_embedding()
  - test_spatio_temporal_embedding_with_masking()
  - test_spatio_temporal_embedding_different_inputs()
  - test_spatio_temporal_embedding_long_sequence()
  
test_video_to_phonemes_pipeline() - Line 547-606
test_spatio_temporal_performance() - Line 621-642
```

**Status:** ~30% of this test file obsolete

---

### 4.4 `/home/user/Valerie/tests/test_model_components.py`

**Obsolete tests:**
```
test_conv3d_block() - Line 40-61
test_temporal_positional_encoding() - Line 78-120
test_spatio_temporal_embedding_basic() - Line 87-127
test_spatio_temporal_embedding_with_padding() - Line 144-169
... and 8+ more spatio-temporal tests
test_spatio_temporal_performance() - Line 263-285
```

**Status:** ~70% of this test file obsolete

---

### 4.5 `/home/user/Valerie/tests/test_components.py`

**Obsolete imports/tests:**
- Imports SpatioTemporalEmbedding (line 24)
- Tests it (line 41-43, 101-120, 147-160)

**Action:** Remove spatio-temporal tests, keep other component tests

---

### 4.6 `/home/user/Valerie/tests/test_dataset_pipeline.py`

**Obsolete sections:**
- VideoTransforms tests (lines 51-68)
- Video batch handling (lines 109-148)
- Video-specific augmentation tests

**Status:** ~40% of this file obsolete

---

## SECTION 5: OUTDATED TRAINING SCRIPTS

### 5.1 OLD TRAINING SCRIPTS (OBSOLETE)

**File:** `/home/user/Valerie/scripts/train.py`

**Status:** OBSOLETE
- Generic training for old ValerieModel (video)
- Not using ensemble distillation
- Uses base_config.yaml (video-based)

---

**File:** `/home/user/Valerie/scripts/train_two_stage.py`

**Status:** OBSOLETE
- Two-stage training: Video→Phonemes, then Phonemes→Text
- Uses `ValerieModel` (video-based)
- Expects video input
- Imports from old pipeline

---

**File:** `/home/user/Valerie/train_multi_gpu.py` (root level)

**Status:** OBSOLETE
- Multi-GPU training for old ValerieModel
- References video processing
- Uses old config system

---

### 5.2 NEW TRAINING SCRIPTS (CORRECT)

These scripts implement the new ensemble distillation approach:

**File:** `/home/user/Valerie/scripts/train_stage1_phoneme_asr.py`
- ✅ Uses `AudioPhonemeASR`
- ✅ Ensemble distillation (3 teachers)
- ✅ LibriSpeech dataset
- ✅ Uses ensemble_distillation_config.yaml

---

**File:** `/home/user/Valerie/scripts/train_stage2_phoneme_to_text.py`
- ✅ Phoneme→Text LLM fine-tuning
- ✅ Text dataset training
- ✅ Synthetic error injection

---

**File:** `/home/user/Valerie/scripts/train_stage3_end_to_end.py`
- ✅ End-to-end fine-tuning
- ✅ Joint training of both stages

---

## SECTION 6: TOOLS STILL USING OBSOLETE MODELS

### 6.1 Inference Engine

**File:** `/home/user/Valerie/src/inference/inference_engine.py`

**Issue:** Imports and uses `ValerieModel` (video-based)
- Should be updated to use `AudioPhonemeASR`
- Expects video input in current form

---

### 6.2 Visualization/Model Analysis

**File:** `/home/user/Valerie/src/visualization/model_analysis.py`

**Issue:** Analyzes `ValerieModel`
- Should analyze `AudioPhonemeASR` instead
- Visualizes spatio-temporal components (now obsolete)

---

### 6.3 Example Scripts

**File:** `/home/user/Valerie/examples/multi_gpu_training_comparison.py`

**Issue:** Compares old multi-GPU training
- Uses `ValerieModel`
- Outdated comparison

---

## SUMMARY TABLE

| Category | File | Lines | Status | Action |
|----------|------|-------|--------|--------|
| **VIDEO 3D CNN** | spatio_temporal.py | 418 | Obsolete | DELETE |
| **VIDEO MODEL** | valerie_model.py | 360 | Redundant | MIGRATE TESTS |
| **VIDEO DATASETS** | dataset.py | 1,317 | Partially Obsolete | REFACTOR |
| **VIDEO AUGMENT** | transforms.py | VideoTransforms part | Obsolete | REMOVE |
| **OLD DISTILL** | audio_distillation.py | 557 | Obsolete | DELETE |
| **NEW DISTILL** | ensemble_distillation.py | 696 | Correct | KEEP |
| **OLD AUDIO** | audio_phoneme_model.py | 456 | Recommended | USE |
| **CONFIG VIDEO** | base_config.yaml | 119 | Deprecated | REPLACE |
| **CONFIG VIDEO** | avspeech_config.yaml | 86 | Deprecated | DELETE |
| **CONFIG VIDEO** | voxceleb2_config.yaml | 66 | Deprecated | DELETE |
| **CONFIG VIDEO** | multi_gpu_config.yaml | 250 | Partially Outdated | UPDATE |
| **CONFIG AUDIO** | ensemble_distillation_config.yaml | 159 | Correct | STANDARD |
| **OLD TRAINING** | train.py | ? | Obsolete | DELETE |
| **OLD TRAINING** | train_two_stage.py | ? | Obsolete | DELETE |
| **OLD TRAINING** | train_multi_gpu.py | 430 | Obsolete | DELETE |

---

## RECOMMENDATIONS

### IMMEDIATE CLEANUP (High Priority)

1. **Delete obsolete files:**
   - `src/models/spatio_temporal.py` (418 lines)
   - `src/models/audio_distillation.py` (557 lines) - Replace uses with ensemble_distillation
   - `configs/base_config.yaml` (video-based)
   - `configs/avspeech_config.yaml`
   - `configs/voxceleb2_config.yaml`
   - `scripts/train.py`
   - `scripts/train_two_stage.py`
   - `train_multi_gpu.py`

2. **Update/Migrate:**
   - `src/models/valerie_model.py` - Not used in new pipeline but still imported
   - `src/data/dataset.py` - Remove video-specific methods, add audio support
   - `src/data/transforms.py` - Remove VideoTransforms class
   - `src/models/__init__.py` - Remove spatio_temporal, audio_distillation exports
   - `configs/multi_gpu_config.yaml` - Use ensemble_distillation_config.yaml

3. **Update Tests:**
   - `test_mandatory_components.py` - Test AudioPhonemeASR instead
   - `test_integration.py` - Remove video tests (30-40%)
   - `test_models.py` - Remove spatio-temporal tests (30%)
   - `test_model_components.py` - Remove spatio-temporal tests (70%)
   - `test_components.py` - Remove spatio-temporal tests
   - `test_dataset_pipeline.py` - Remove VideoTransforms tests (40%)

4. **Update Tools:**
   - `src/inference/inference_engine.py` - Use AudioPhonemeASR
   - `src/visualization/model_analysis.py` - Analyze new models
   - `examples/multi_gpu_training_comparison.py` - Update or delete

### MEDIUM PRIORITY

- Consolidate duplicated distillation code
- Create audio-only dataset loaders
- Standardize on ensemble_distillation_config.yaml
- Update README.md to remove video references

### DEFERRED (Low Priority but Good to Do)

- Archive old configs for reference
- Document migration path from video to audio
- Create migration guide for users

