import pandas as pd
import json
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT_DIR / "data" / "flickr1k_small" / "captions.csv"
JSON_OUTPUT = ROOT_DIR / "data" / "flickr1k_small" / "captions.json"

# Load CSV
df = pd.read_csv(CSV_PATH)

# Make sure the CSV has the expected columns
# Usually: image, caption
if not {"image", "caption"}.issubset(df.columns):
    raise ValueError(f"CSV must contain 'image' and 'caption' columns, found: {df.columns.tolist()}")

# Group captions by image
captions_dict = {}
for _, row in df.iterrows():
    img = row["image"]
    cap = row["caption"]
    captions_dict.setdefault(img, []).append(cap)

# Save JSON
with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(captions_dict, f, indent=4)

print(f"Converted {len(df)} rows to JSON at {JSON_OUTPUT}")
