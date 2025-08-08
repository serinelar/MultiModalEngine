import json

CAPTIONS_FILE = "data/flickr_captions.json"

def load_captions(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def search_captions(captions: dict, keyword: str) -> dict:
    keyword = keyword.lower()
    result = {}

    for image, caption_list in captions.items():
        for caption in caption_list:
            if keyword in caption.lower():
                if image not in result:
                    result[image] = []
                result[image].append(caption)

    return result

if __name__ == "__main__":
    captions = load_captions(CAPTIONS_FILE)
    
    for image_name, image_captions in list(captions.items())[:1]:
        print(f"Image: {image_name}")
        for i, caption in enumerate(image_captions, 1):
            print(f"  Caption {i}: {caption}")

    searched = search_captions(captions, 'two')
    for image, matched_captions in searched.items():
        print(f"\nImage: {image}")
        for i, caption in enumerate(matched_captions, 1):
            print(f"  Match {i}: {caption}")
