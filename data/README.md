---
title: VoxCeleb2 Dataset
license: mit
dataset_info:
  features:
  - name: audio
    dtype: audio
  - name: speaker_id
    dtype: string
  - name: video_id
    dtype: string
task_categories:
- automatic-speech-recognition
- speaker-verification
- audio-classification
language:
- en
pretty_name: VoxCeleb2 Dataset
size_categories:
- 100K<n<1M
---

# VoxCeleb2 Dataset

This is the VoxCeleb2 dataset, a large-scale speaker identification dataset.

## Dataset Description

VoxCeleb2 contains over 1 million utterances for 6,112 celebrities, extracted from videos uploaded to YouTube.

## Files

- `vox2_dev_mp4_part*`: Multipart archive containing MP4 video files
- `vox2_dev_txt`: Text files with speaker/utterance metadata  
- `vox2_meta.csv`: Dataset metadata

## Usage

To extract the multipart archive:
```bash
# Using 7zip
7z x vox2_dev_mp4_partaa

# Or using cat (Linux/Mac)
cat vox2_dev_mp4_parta* > combined.tar
tar -xf combined.tar
```

## Citation

If you use this dataset, please cite:
```
@inproceedings{chung2018voxceleb2,
  title={VoxCeleb2: Deep Speaker Recognition},
  author={Chung, Joon Son and Nagrani, Arsha and Zisserman, Andrew},
  booktitle={INTERSPEECH},
  year={2018}
}
```

## License

Please refer to the original VoxCeleb2 license terms.
