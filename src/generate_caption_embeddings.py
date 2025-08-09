import json
import pickle
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import spacy
from pathlib import Path

# Load models
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Get root dir (1 level up from current file location)
ROOT_DIR = Path(__file__).resolve().parents[1]

# Paths
CAPTIONS_FILE = ROOT_DIR / "data" / "flickr_captions.json"
PICKLE_OUTPUT = ROOT_DIR / "data" / "captions_data.pkl"
EMBEDDINGS_OUTPUT = ROOT_DIR / "data" / "caption_embeddings.npy"

def load_captions(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def preprocess_captions(captions_dict):
    original_captions = []
    cleaned_captions = []

    for image, caption_list in captions_dict.items():
        for caption in caption_list:
            original_captions.append(caption)
            doc = nlp(caption.lower())
            lemmas = [token.lemma_ for token in doc]
            cleaned = " ".join(lemmas)
            cleaned_captions.append(cleaned)

    return {
        "original_captions": original_captions,
        "cleaned_captions": cleaned_captions
    }

if __name__ == "__main__":
    print("Loading captions...")
    captions_dict = load_captions(CAPTIONS_FILE)

    print("Preprocessing captions...")
    captions_data = preprocess_captions(captions_dict)

    print(f"Saving cleaned captions to {PICKLE_OUTPUT}...")
    with open(PICKLE_OUTPUT, "wb") as f:
        pickle.dump(captions_data, f)

    print("Generating embeddings...")
    embeddings = model.encode(
        captions_data["cleaned_captions"],
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=64
    )

    print(f"Saving embeddings to {EMBEDDINGS_OUTPUT}...")
    np.save(EMBEDDINGS_OUTPUT, embeddings)

    print(f"Done! Total captions: {len(captions_data['original_captions'])}")
