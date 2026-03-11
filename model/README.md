# Model Files

The trained model weights are not included in this repository due to file size (~109 MB total).

## Required files

| File | Description | Size |
|------|-------------|------|
| `Audio_SilentCare_model.h5` | Audio classification head (Keras, YAMNet embeddings) | 9.4 MB |
| `audio_silentcare_classes.npy` | Class label mapping | 256 B |
| `Video_SilentCare_model.pth` | ResNet50 fine-tuned on RAF-DB | 94 MB |

## Download pretrained models

```bash
python download_models.py
```

> Update the placeholder URLs in `download_models.py` after uploading the weights to GitHub Releases.

## Retrain from scratch

```bash
python -m silentcare.training.train_audio
python -m silentcare.training.train_video
```

Requires the RAF-DB dataset for video and the prepared audio dataset (see README.md).
