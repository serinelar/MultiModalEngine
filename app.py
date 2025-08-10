import streamlit as st
from pathlib import Path
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration
import spacy
import io
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]

DATASETS = {
    "flickr1k_small": {
        "captions_json": "data/flickr1k_small/captions.json",
        "caption_embeddings": "data/flickr1k_small/caption_embeddings.npy",
        "image_embeddings": "data/flickr1k_small/image_embeddings.npy",
        "image_filenames": "data/flickr1k_small/image_filenames.npy",
        "images_dir": "data/flickr1k_small/images"
    },
    # Add additional dataset entries here if/when available
}

# Caching: load heavy models/resources once
@st.cache_resource(show_spinner=False)
def load_models():
    txt_model = SentenceTransformer("all-MiniLM-L6-v2")
    clip_model = SentenceTransformer("clip-ViT-B-32")
    # BLIP (caption generator)
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    # SpaCy (lightweight)
    nlp = spacy.load("en_core_web_sm")
    return txt_model, clip_model, blip_processor, blip_model, nlp

@st.cache_data(show_spinner=False)
def load_embeddings_and_files(dataset_key):
    info = DATASETS[dataset_key]
    cap_emb = np.load(info["caption_embeddings"]).astype("float32")
    img_emb = np.load(info["image_embeddings"]).astype("float32")
    img_files = np.load(info["image_filenames"], allow_pickle=True)
    # Normalize so cosine similarity = dot product
    def l2_norm_rows(x):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return x / norms
    cap_emb = l2_norm_rows(cap_emb)
    img_emb = l2_norm_rows(img_emb)
    return cap_emb, img_emb, img_files, info["images_dir"], info["captions_json"]

# Utility helpers
def flatten_captions_dict(captions_dict):
    filenames = []
    orig_captions = []
    for filename, c_list in captions_dict.items():
        for c in c_list:
            filenames.append(filename)
            orig_captions.append(c)
    return filenames, orig_captions

def aggregate_caption_hits(top_idx, top_scores, filenames_list, orig_captions_list, top_images=8):
    # Aggregate top caption hits to image-level results (max strategy)
    image_hits = {}
    for idx, score in zip(top_idx, top_scores):
        fn = filenames_list[idx]
        cap = orig_captions_list[idx]
        image_hits.setdefault(fn, []).append((float(score), cap, int(idx)))
    image_list = []
    for fn, hits in image_hits.items():
        image_list.append((fn, max(h[0] for h in hits), hits))
    image_list.sort(key=lambda x: x[1], reverse=True)
    results = []
    for fn, score, hits in image_list[:top_images]:
        results.append({
            "filename": fn,
            "score": score,
            "matches": [{"caption": h[1], "score": h[0], "caption_idx": h[2]} for h in hits]
        })
    return results

def read_image_from_uploaded(uploaded_file):
    # streamlit provides an UploadedFile; convert to PIL Image
    image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
    return image

# Streamlit UI
st.set_page_config(page_title="Multimodal Search (Demo)", layout="wide")
st.title("Multimodal Search — Demo (Text / Image / Caption)")

# Sidebar: dataset selector + settings
st.sidebar.header("Settings")
dataset_key = st.sidebar.selectbox("Dataset", list(DATASETS.keys()))
cap_emb, img_emb, img_files, images_dir, captions_json = load_embeddings_and_files(dataset_key)
txt_model, clip_model, blip_processor, blip_model, nlp = load_models()

# Load captions dict (small, OK to load)
import json
with open(captions_json, "r", encoding="utf-8") as f:
    captions_dict = json.load(f)
filenames_list, orig_captions_list = flatten_captions_dict(captions_dict)

tab1, tab2, tab3 = st.tabs(["Text → Image", "Image → Image", "Image → Caption"])

# Tab 1: Text -> Image
with tab1:
    st.header("Text → Image")
    query = st.text_input("Enter text query", value="a woman riding a horse")
    top_k = st.slider("Number of caption hits to consider (k)", min_value=10, max_value=500, value=200, step=10)
    top_images = st.slider("Top images to display", min_value=1, max_value=12, value=6)
    if st.button("Search (Text → Image)"):
        # encode query (we keep it simple: lowercase + lemmatize to match preprocess if you used it)
        doc = nlp(query.lower())
        query_cleaned = " ".join([t.lemma_ for t in doc])
        qvec = txt_model.encode([query_cleaned], convert_to_numpy=True)[0].astype("float32")
        qvec /= (np.linalg.norm(qvec) + 1e-10)
        sims = cap_emb.dot(qvec)  # shape (num_captions,)
        # top k captions indices
        if top_k >= sims.shape[0]:
            top_idx = np.argsort(-sims)
        else:
            top_idx = np.argpartition(-sims, range(top_k))[:top_k]
            top_idx = top_idx[np.argsort(-sims[top_idx])]
        top_scores = sims[top_idx]
        # aggregate to images
        results = aggregate_caption_hits(top_idx, top_scores, filenames_list, orig_captions_list, top_images=top_images)
        # Display
        cols = st.columns(min(top_images, 6))
        for i, res in enumerate(results):
            col = cols[i % len(cols)]
            img_path = os.path.join(images_dir, res["filename"])
            col.image(str(img_path), use_column_width=True, caption=f"{res['filename']} ({res['score']:.3f})")
            for m in res["matches"][:3]:
                col.write(f"- [{m['score']:.3f}] {m['caption']}")

# Tab 2: Image -> Image
with tab2:
    st.header("Image → Image (upload a query image)")
    uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])
    top_k_img = st.slider("Top similar images to return", 1, 20, 6)
    if uploaded is not None:
        query_img = read_image_from_uploaded(uploaded)
        st.image(query_img, caption="Query image", use_column_width=False, width=250)
        # encode with CLIP model (SentenceTransformer wrapper accepts PIL images)
        qvec = clip_model.encode([query_img], convert_to_numpy=True, normalize_embeddings=True)[0].astype("float32")
        sims = img_emb.dot(qvec)  # dot product since normalized
        top_idx = np.argsort(-sims)[:top_k_img]
        results = [{"filename": img_files[i], "score": float(sims[i])} for i in top_idx]
        cols = st.columns(min(top_k_img, 6))
        for i, r in enumerate(results):
            col = cols[i % len(cols)]
            col.image(str(Path(images_dir) / r["filename"]), use_column_width=True, caption=f"{r['filename']} ({r['score']:.3f})")

# Tab 3: Image -> Caption
with tab3:
    st.header("Image → Caption (caption generation using BLIP)")
    uploaded2 = st.file_uploader("Upload an image for captioning", type=["jpg", "jpeg", "png"], key="cap")
    if uploaded2 is not None:
        query_img = read_image_from_uploaded(uploaded2)
        st.image(query_img, caption="Image to caption", use_column_width=False, width=300)
        # BLIP processing
        inputs = blip_processor(images=query_img, return_tensors="pt")
        out = blip_model.generate(**inputs)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        st.subheader("Generated caption")
        st.write(caption)

# Footer / Notes
st.sidebar.markdown("---")
st.sidebar.markdown("Tip: Use the small dataset for smooth local testing.")
