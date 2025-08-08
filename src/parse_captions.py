import json

def parse_flickr30k_captions(filepath):
    """
    Parses Flickr30k JSON into a dictionary {image_filename: [captions]}
    Args:
        filepath (str): Path to dataset_flickr30k.json

    Returns:
        dict: A dictionary mapping image filenames to a list of raw captions.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    parsed = {}
    for image_data in data['images']:
        filename = image_data['filename']
        captions = [s['raw'] for s in image_data['sentences']]
        parsed[filename] = captions

    return parsed
