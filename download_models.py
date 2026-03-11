"""Download pretrained SilentCare model weights from GitHub Releases."""

import urllib.request
import os

MODELS = {
    "model/Audio_SilentCare_model.h5": "PLACEHOLDER_URL_AUDIO",
    "model/audio_silentcare_classes.npy": "PLACEHOLDER_URL_CLASSES",
    "model/Video_SilentCare_model.pth": "PLACEHOLDER_URL_VIDEO",
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
