# Test Migration Status

## ✅ Updated Tests (Audio-Only)

- **test_mandatory_components.py** - Fully migrated to AudioPhonemeASR (Phase 3)
- **test_models.py** - Fully migrated to audio-only architecture (Phase 5)
- **test_integration.py** - Fully migrated to audio-only tests (Phase 5)

## ⚠️ Tests Needing Migration (Video-Based)

The following tests still use video-based components and need migration:

### test_model_components.py
- 70% of tests are spatio-temporal (obsolete)
- Tests Conv3dBlock, TemporalPositionalEncoding, etc.
- **Action:** Remove video component tests, keep Conformer/CTC tests
- **Priority:** Medium

### test_components.py
- Imports SpatioTemporalEmbedding (obsolete)
- Tests video components
- **Action:** Remove spatio-temporal imports/tests
- **Priority:** Low

### test_dataset_pipeline.py
- 40% VideoTransforms tests (obsolete)
- Video batch handling
- **Action:** Remove VideoTransforms tests, add audio tests
- **Priority:** Medium

## Migration Guide

To update a test file:

1. Replace `from src.models.valerie_model import ValerieModel` 
   with `from src.models.audio_phoneme_model import AudioPhonemeASR`

2. Replace video inputs:
   ```python
   # Old (video)
   video = torch.randn(batch_size, 3, 8, 224, 224)  # [B, C, T, H, W]
   outputs = model(video, input_lengths)
   
   # New (audio)
   audio = torch.randn(batch_size, 16000)  # [B, T_audio]
   outputs = model(audio=audio, audio_lengths=audio_lengths)
   ```

3. Remove spatio-temporal component tests (Conv3dBlock, TemporalPositionalEncoding, etc.)

4. Remove VideoTransforms tests

5. Update imports to use new models (AudioFrontend, AudioPhonemeASR, EnsembleDistillationModule)

## Test Priority

**✅ High Priority (COMPLETED):**
- ✅ test_models.py - Core model tests (Phase 5)
- ✅ test_integration.py - End-to-end pipeline (Phase 5)

**Medium Priority:**
- test_model_components.py - Component unit tests
- test_dataset_pipeline.py - Data pipeline

**Low Priority:**
- test_components.py - Individual component tests

