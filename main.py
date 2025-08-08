import json
import spacy
nlp = spacy.load("en_core_web_sm")

CAPTIONS_FILE = "data/flickr_captions.json"

def load_captions(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
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

if __name__ == "__main__":
    captions = dict(list(load_captions(CAPTIONS_FILE).items())[:10])
    processed = preprocess_captions(captions)

#    for image_name, image_captions in list(captions.items())[:1]:
#       print(f"Image: {image_name}")
#        for i, caption in enumerate(image_captions, 1):
#            print(f"  Caption {i}: {caption}")

    searched = search_captions(processed, 'MEN')
    for image, matched_captions in searched.items():
        print(f"\nImage: {image}")
        for i, caption in enumerate(matched_captions, 1):
            print(f"  Match {i}: {caption}")
