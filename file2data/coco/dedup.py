import json
import argparse
from typing import Dict, List
from PIL import Image
import imagehash

def dedup_images(coco_file: str, output_file: str, method: str = 'file_name') -> None:
    """
    Deduplicate images in a COCO JSON file.

    Args:
        coco_file (str): Path to the COCO JSON file.
        output_file (str): Path to the output deduplicated COCO JSON file.
        method (str): Method to use for deduplication ('file_name' or 'phash').
    """
    with open(coco_file, 'r') as f:
        coco = json.load(f)

    if method == 'file_name':
        unique_images = []
        unique_image_ids = set()
        for img in coco['images']:
            if img['file_name'] not in unique_image_ids:
                unique_images.append(img)
                unique_image_ids.add(img['file_name'])

    elif method == 'phash':
        unique_images = []
        unique_hashes = set()
        for img in coco['images']:
            image_path = img['file_name']
            image = Image.open(image_path)
            phash = imagehash.phash(image)
            if phash not in unique_hashes:
                unique_images.append(img)
                unique_hashes.add(phash)

    else:
        raise ValueError(f"Unsupported deduplication method: {method}")

    unique_image_ids = {img['id'] for img in unique_images}
    unique_annotations = [ann for ann in coco['annotations'] if ann['image_id'] in unique_image_ids]

    deduped_coco = {
        'images': unique_images,
        'annotations': unique_annotations,
        'categories': coco['categories']
    }

    with open(output_file, 'w') as f:
        json.dump(deduped_coco, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deduplicate images in a COCO JSON file.")
    parser.add_argument("coco_file", type=str, help="Path to the COCO JSON file")
    parser.add_argument("output_file", type=str, help="Path to the output deduplicated COCO JSON file")
    parser.add_argument("--method", type=str, choices=['file_name', 'phash'], default='file_name', help="Method to use for deduplication ('file_name' or 'phash')")
    args = parser.parse_args()

    dedup_images(args.coco_file, args.output_file, args.method)