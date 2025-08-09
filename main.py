import json
import spacy
from pathlib import Path
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util

nlp = spacy.load("en_core_web_sm")

# Project-root-aware paths (safe whether you run from src/ or project root)
ROOT_DIR = Path(__file__).resolve().parents[1]
CAPTIONS_JSON = "data/flickr_captions.json"
CAPTIONS_PICKLE = "data/captions_data.pkl"
EMBEDDINGS_NPY = "data/caption_embeddings.npy"

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_captions(CAPTIONS_JSON):
    with open(CAPTIONS_JSON, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def preprocess_captions(captions: dict) -> dict:
    processed = {}
    for image, caption_list in captions.items():
        processed[image] = []
        for caption in caption_list:
            doc = nlp(caption.lower())
            lemmas = [token.lemma_ for token in doc]
            processed[image].append((caption, lemmas))
    return processed

def search_captions(captions: dict, keyword: str) -> dict:
    keyword = keyword.lower()
    keyword_lemma = nlp(keyword)[0].lemma_  # e.g., "men" → "man"
    result = {}

    for image, caption_list in captions.items():
        for original_caption, lemmas in caption_list:
            if keyword_lemma in lemmas:
                result.setdefault(image, []).append(original_caption)
    return result

def flatten_captions_dict(captions_dict):
    """
    Convert captions dict {filename: [cap1, cap2]} into two parallel lists
    (filenames_list, orig_captions_list) with the same order used during embedding creation.
    """
    filenames = []
    orig_captions = []
    for filename, c_list in captions_dict.items():
        for c in c_list:
            filenames.append(filename)
            orig_captions.append(c)
    return filenames, orig_captions

def l2_normalize_rows(x: np.ndarray):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return x / norms

def semantic_search_query(query: str, embeddings: np.ndarray, captions_filenames, orig_captions,
                          top_k=50, top_images=10):
    """
    1) preprocess query with spaCy (same as during embedding creation)
    2) encode query with the same SentenceTransformer
    3) compute cosine similarity via dot product on normalized vectors
    4) return aggregated top images (max caption score per image)
    """
    # 1. preprocess query (lemmatize)
    doc = nlp(query.lower())
    query_cleaned = " ".join([t.lemma_ for t in doc])

    # 2. encode query (returns 1 x D)
    qvec = model.encode([query_cleaned], convert_to_numpy=True)[0].astype("float32")
    qvec = qvec / (np.linalg.norm(qvec) + 1e-10)  # normalize to unit length

    # 3. compute cosine similarities (embeddings must be normalized already)
    sims = embeddings.dot(qvec)  # shape: (N,)

    # 4. top-k caption indices (fast)
    if top_k >= sims.shape[0]:
        top_idx = np.argsort(-sims)
    else:
        top_idx = np.argpartition(-sims, range(top_k))[:top_k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

    top_scores = sims[top_idx]

    # 5. aggregate per image (max strategy)
    image_hits = {}
    for idx, score in zip(top_idx, top_scores):
        fn = captions_filenames[idx]
        cap = orig_captions[idx]
        image_hits.setdefault(fn, []).append((float(score), cap, int(idx)))

    # compute image-level score = max caption score
    image_list = []
    for fn, hits in image_hits.items():
        max_score = max(h[0] for h in hits)
        image_list.append((fn, max_score, hits))

    image_list.sort(key=lambda x: x[1], reverse=True)
    # limit
    results = []
    for fn, score, hits in image_list[:top_images]:
        results.append({
            "filename": fn,
            "score": score,
            "matches": [{"caption": h[1], "score": h[0], "caption_idx": h[2]} for h in hits]
        })
    return results

if __name__ == "__main__":
#    captions = dict(list(load_captions(CAPTIONS_FILE).items())[:10])
#    processed = preprocess_captions(captions)

#    for image_name, image_captions in list(captions.items())[:1]:
#       print(f"Image: {image_name}")
#        for i, caption in enumerate(image_captions, 1):
#            print(f"  Caption {i}: {caption}")

#    searched = search_captions(processed, 'MEN')
#    for image, matched_captions in searched.items():
#        print(f"\nImage: {image}")
#        for i, caption in enumerate(matched_captions, 1):
#            print(f"  Match {i}: {caption}")

    captions_dict = load_captions(str(CAPTIONS_JSON))

    filenames_list, orig_captions_list = flatten_captions_dict(captions_dict)

    embeddings = np.load(str(EMBEDDINGS_NPY)).astype("float32")
    embeddings = l2_normalize_rows(embeddings)  # normalize once

    if len(orig_captions_list) != embeddings.shape[0]:
        raise RuntimeError(f"Length mismatch: captions={len(orig_captions_list)} embeddings={embeddings.shape[0]}")

    query = "WOMAN AND A CHILD"   # change to test
    print(f"\nSearching for: {query}")
    results = semantic_search_query(query, embeddings, filenames_list, orig_captions_list,
                                    top_k=200, top_images=8)

    for r in results:
        print(f"\nImage: {r['filename']}  (score={r['score']:.4f})")
        for m in r['matches']:
            print(f"    • [{m['score']:.4f}] {m['caption']}")