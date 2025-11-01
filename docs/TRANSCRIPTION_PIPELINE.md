# 🎤 Transcription Pipeline for VALLR-Style Training

## 🚨 **The Problem We Solved**

VoxCeleb2 dataset has a **critical missing component**:

```yaml
❌ What VoxCeleb2 Actually Contains:
  - video.mp4: ✅ Video with embedded audio
  - audio.aac: ✅ Separate audio files  
  - text.txt:  ❌ Face bounding boxes (NOT transcriptions!)

✅ What VALLR Training Needs:
  - video.mp4: ✅ Video files
  - transcriptions.txt: ❌ Speech transcriptions (missing!)
  - phonemes.txt: ❌ Phoneme sequences (missing!)
```

### 📊 **VoxCeleb2 Text File Format (NOT Transcriptions)**

```
Identity  : 	id00012
Reference : 	_raOc3-IRsw
Offset    : 	2
FV Conf   : 	20.230	(1)
ASD Conf  : 	2.765

FRAME 	X 	Y 	W 	H 
000349 	0.319 	0.067 	0.424 	0.753 
000350 	0.319 	0.067 	0.424 	0.753 
...
```

This is **face detection data**, not speech transcriptions!

## 🎯 **Our Solution: ASR-Based Transcription Pipeline**

### **Phase 1: Generate Transcriptions**
- Extract audio from MP4 files (or use existing AAC files)
- Use OpenAI Whisper Large-v3 for high-quality transcriptions
- Generate word-level timestamps and confidence scores

### **Phase 2: Convert to Phonemes**
- Convert transcriptions to phoneme sequences
- Support multiple phoneme sets (ARPABET, IPA)
- Generate phoneme-to-frame alignments

### **Phase 3: VALLR-Style Training**
- Stage 1: Video → Phonemes (CTC loss)
- Stage 2: Phonemes → Text (LLM fine-tuning)

## 🚀 **Quick Start**

### **1. Install Dependencies**

```bash
# Install transcription requirements
pip install -r requirements_transcription.txt

# Optional: Install eSpeak for better phoneme conversion
# Ubuntu/Debian:
sudo apt-get install espeak-ng
# macOS:
brew install espeak
# Windows: Download from http://espeak.sourceforge.net/
```

### **2. Test the Pipeline**

```bash
# Test with synthetic data
python scripts/test_transcription_pipeline.py --create_sample

# Test with your own file
python scripts/test_transcription_pipeline.py --test_file path/to/video.mp4
```

### **3. Generate Transcriptions**

```bash
# Generate transcriptions for your VoxCeleb2 data
python scripts/generate_transcriptions.py \
    --data_root data/voxceleb2 \
    --output_dir data/transcriptions \
    --model large-v3 \
    --max_files 100  # For testing, remove for full dataset

# Monitor progress
tail -f transcription_generation.log
```

### **4. Convert to Phonemes**

```bash
# Convert transcriptions to phonemes
python scripts/convert_to_phonemes.py \
    --transcription_dir data/transcriptions \
    --output_dir data/phonemes \
    --phoneme_set arpabet

# Check results
ls data/phonemes/
```

## 📊 **Pipeline Performance**

### **Whisper Model Comparison**

| Model | Parameters | VRAM | Speed | WER | Best For |
|-------|------------|------|--------|-----|----------|
| **tiny** | 39M | ~1GB | 32x | ~5-10% | Testing, fast processing |
| **base** | 74M | ~1GB | 16x | ~3-7% | Good balance |
| **small** | 244M | ~2GB | 6x | ~2-5% | Better accuracy |
| **medium** | 769M | ~5GB | 2x | ~1-4% | High accuracy |
| **large-v3** | 1.55B | ~10GB | 1x | ~1-3% | **Best accuracy** |

### **Processing Estimates (VoxCeleb2)**

```yaml
Dataset Size: ~2,300 hours of video
File Count: ~150,000 MP4 files

Whisper tiny (recommended for testing):
  - Speed: ~10 files/minute
  - Total time: ~250 hours (~10 days)
  - Cost (RTX 4090): ~$300

Whisper large-v3 (recommended for production):
  - Speed: ~2 files/minute  
  - Total time: ~1,250 hours (~52 days)
  - Cost (RTX 4090): ~$1,500
```

### **Optimization Strategies**

#### **1. Batch Processing**
```bash
# Process in smaller batches
python scripts/generate_transcriptions.py \
    --data_root data/voxceleb2 \
    --output_dir data/transcriptions \
    --max_files 1000 \
    --model base  # Faster model for initial processing
```

#### **2. GPU Memory Management**
```bash
# Single-threaded to avoid OOM
python scripts/generate_transcriptions.py \
    --num_workers 1 \
    --model large-v3
```

#### **3. Resume Processing**
```bash
# Pipeline automatically skips existing files
# Just re-run the same command to resume
python scripts/generate_transcriptions.py \
    --data_root data/voxceleb2 \
    --output_dir data/transcriptions
```

## 📁 **Output Structure**

### **Generated Transcriptions**

```
data/transcriptions/
├── id00012_raOc3-IRsw_00001.json     # Detailed transcription data
├── id00012_raOc3-IRsw_00001.txt      # Simple text transcription
├── id00013_4PLhpI2ONQE_00002.json
├── id00013_4PLhpI2ONQE_00002.txt
└── transcription_stats.json          # Processing statistics
```

#### **JSON Format**
```json
{
  "file": "/path/to/video.mp4",
  "audio_source": "/path/to/audio.aac",
  "transcription": {
    "text": "Hello this is a test transcription",
    "language": "en",
    "duration": 3.2,
    "words": [
      {"word": "Hello", "start": 0.0, "end": 0.5, "probability": 0.99},
      {"word": "this", "start": 0.6, "end": 0.8, "probability": 0.98}
    ]
  },
  "processing_time": 2.1,
  "model": "large-v3",
  "status": "success"
}
```

### **Generated Phonemes**

```
data/phonemes/
├── id00012_raOc3-IRsw_00001.json     # Detailed phoneme data
├── id00012_raOc3-IRsw_00001.txt      # Simple phoneme sequence
├── id00013_4PLhpI2ONQE_00002.json
├── id00013_4PLhpI2ONQE_00002.txt
└── phoneme_conversion_stats.json     # Conversion statistics
```

#### **Phoneme JSON Format**
```json
{
  "file": "/path/to/transcription.json",
  "original_text": "Hello this is a test",
  "phonemes": ["HH", "AH", "L", "OW", "SIL", "DH", "IH", "S", "SIL", "IH", "Z", "SIL", "AH", "SIL", "T", "EH", "S", "T"],
  "phoneme_count": 18,
  "phoneme_set": "arpabet",
  "word_count": 5,
  "status": "success"
}
```

## 🔧 **Integration with Training**

### **Updated Data Loader**

```python
class VoxCeleb2Dataset(Dataset):
    """VoxCeleb2 dataset with generated transcriptions and phonemes."""
    
    def __init__(self, video_dir, transcription_dir, phoneme_dir):
        self.video_dir = Path(video_dir)
        self.transcription_dir = Path(transcription_dir) 
        self.phoneme_dir = Path(phoneme_dir)
        
        # Find matching triplets (video, transcription, phonemes)
        self.samples = self._find_matching_samples()
    
    def _find_matching_samples(self):
        samples = []
        for video_path in self.video_dir.rglob("*.mp4"):
            base_name = video_path.stem
            
            transcription_path = self.transcription_dir / f"{base_name}.json"
            phoneme_path = self.phoneme_dir / f"{base_name}.json"
            
            if transcription_path.exists() and phoneme_path.exists():
                samples.append({
                    "video": video_path,
                    "transcription": transcription_path,
                    "phonemes": phoneme_path
                })
        
        return samples
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load video
        video = self.load_video(sample["video"])
        
        # Load transcription
        with open(sample["transcription"]) as f:
            transcription_data = json.load(f)
        text = transcription_data["transcription"]["text"]
        
        # Load phonemes
        with open(sample["phonemes"]) as f:
            phoneme_data = json.load(f)
        phonemes = phoneme_data["phonemes"]
        
        return {
            "video": video,
            "text": text,
            "phonemes": phonemes,
            "video_path": str(sample["video"])
        }
```

### **VALLR-Style Training Loop**

```python
# Stage 1: Video → Phonemes (CTC only)
def train_stage1():
    for batch in dataloader:
        video = batch["video"]
        target_phonemes = batch["phonemes"]
        
        # Forward pass (visual encoder only)
        visual_features = model.visual_encoder(video)
        phoneme_logits = model.ctc_head(visual_features)
        
        # CTC loss only
        loss = ctc_loss(phoneme_logits, target_phonemes)
        loss.backward()
        optimizer.step()

# Stage 2: Phonemes → Text (LLM only)
def train_stage2():
    for batch in dataloader:
        phonemes = batch["phonemes"]
        target_text = batch["text"]
        
        # Forward pass (LLM only)
        predicted_text = model.phoneme_to_text(phonemes)
        
        # Language modeling loss only
        loss = lm_loss(predicted_text, target_text)
        loss.backward()
        optimizer.step()
```

## 🎯 **Next Steps**

### **Immediate Actions**

1. **Test the pipeline**:
   ```bash
   python scripts/test_transcription_pipeline.py --create_sample
   ```

2. **Process a small subset**:
   ```bash
   python scripts/generate_transcriptions.py \
       --data_root data/voxceleb2 \
       --output_dir data/transcriptions \
       --max_files 50 \
       --model base
   ```

3. **Convert to phonemes**:
   ```bash
   python scripts/convert_to_phonemes.py \
       --transcription_dir data/transcriptions \
       --output_dir data/phonemes
   ```

4. **Update training pipeline** to use generated data

### **Production Deployment**

1. **Full dataset processing** (estimate: 10-50 days depending on model)
2. **Quality validation** of generated transcriptions
3. **VALLR-style two-stage training** implementation
4. **Performance evaluation** against current approach

## 🚨 **Important Notes**

### **Quality Considerations**

- **Whisper accuracy**: ~1-3% WER on clean speech
- **Noisy audio**: May have higher error rates
- **Multiple speakers**: Whisper handles this reasonably well
- **Background music**: May affect transcription quality

### **Storage Requirements**

```yaml
Original VoxCeleb2: ~500 GB
Generated Transcriptions: ~2 GB
Generated Phonemes: ~1 GB
Total Additional: ~3 GB (0.6% increase)
```

### **Cost Analysis**

```yaml
Current Approach (End-to-End):
  - Training time: 8.8 days
  - Cost: $253
  - VRAM: 19.3 GB (doesn't fit RTX 4090)

VALLR Approach (Two-Stage):
  - Transcription generation: $300-1,500 (one-time)
  - Stage 1 training: 3 days, $86
  - Stage 2 training: 2 days, $58
  - Total training cost: $144
  - VRAM: 4.8-8.5 GB (fits RTX 4090)

Total Savings: $109 in training costs + better GPU compatibility
```

This pipeline solves the fundamental data problem and enables VALLR-style training with significant cost and efficiency benefits! 🚀
