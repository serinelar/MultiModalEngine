import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parents[1]  # Project root

# Toggle here to switch dataset
USE_SMALL_DATASET = True

if USE_SMALL_DATASET:
    IMAGES_DIR = ROOT_DIR / "data" / "flickr1k_small" / "images"
    OUTPUT_NPY = ROOT_DIR / "data" / "flickr1k_small" / "image_embeddings.npy"
    OUTPUT_FILENAMES = ROOT_DIR / "data" / "flickr1k_small" / "image_filenames.npy"
else:
    IMAGES_DIR = ROOT_DIR / "data" / "flickr_images"
    OUTPUT_NPY = ROOT_DIR / "data" / "image_embeddings.npy"
    OUTPUT_FILENAMES = ROOT_DIR / "data" / "image_filenames.npy"

# Model: CLIP (image+text in same vector space)
model = SentenceTransformer("clip-ViT-B-32")

image_paths = []
for ext in ("*.jpg", "*.jpeg", "*.png"):
    image_paths.extend(Path(IMAGES_DIR).glob(ext))

if not image_paths:
    raise FileNotFoundError(f"No images found in {IMAGES_DIR}")

print(f"Found {len(image_paths)} images in {IMAGES_DIR}")

all_embeddings = []
all_filenames = []

print("Generating image embeddings...")

batch_size = 32
for i in tqdm(range(0, len(image_paths), batch_size)):
    batch_paths = image_paths[i:i+batch_size]

    images = []
    for p in batch_paths:
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
            all_filenames.append(str(p.name))
        except Exception as e:
            print(f"Error loading {p}: {e}")

    if images:
        emb = model.encode(
            images,
            batch_size=len(images),
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        all_embeddings.append(emb)

all_embeddings = np.vstack(all_embeddings).astype("float32")

OUTPUT_NPY.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

np.save(OUTPUT_NPY, all_embeddings)
np.save(OUTPUT_FILENAMES, np.array(all_filenames))

print(f"Saved {all_embeddings.shape[0]} image embeddings to {OUTPUT_NPY}")
print(f"Saved filenames to {OUTPUT_FILENAMES}")