"""Download pretrained SilentCare model weights from GitHub Releases."""

import urllib.request
import os

MODELS = {
    "model/Audio_SilentCare_model.h5": "https://github.com/tomscsr/NiTech_SilentCare/releases/download/v1.0.0/Audio_SilentCare_model.h5",
    "model/audio_silentcare_classes.npy": "https://github.com/tomscsr/NiTech_SilentCare/releases/download/v1.0.0/audio_silentcare_classes.npy",
    "model/Video_SilentCare_model.pth": "https://github.com/tomscsr/NiTech_SilentCare/releases/download/v1.0.0/Video_SilentCare_model.pth",
}

os.makedirs("model", exist_ok=True)

for path, url in MODELS.items():
    if os.path.exists(path):
        print(f"Already exists: {path}")
        continue
    if url.startswith("PLACEHOLDER"):
        print(f"Skipping {path} -- update the URL in this script after uploading to GitHub Releases")
        continue
    print(f"Downloading {path}...")
    urllib.request.urlretrieve(url, path)
    print(f"Done: {path}")
