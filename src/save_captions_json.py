from parse_captions import parse_flickr30k_captions
import json

if __name__ == "__main__":
    input_path = "data/dataset_flickr30k.json"
    output_path = "data/flickr_captions.json"

    captions_dict = parse_flickr30k_captions(input_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(captions_dict, f, indent=2)

    print(f"Saved cleaned captions to {output_path}")
