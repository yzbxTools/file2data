import json
import argparse
import random
from typing import Dict, List

def sample_categories(coco_file: str, output_file: str, sample_size: int) -> None:
    """
    Sample categories with related images and annotations from a COCO JSON file.

    Args:
        coco_file (str): Path to the COCO JSON file.
        output_file (str): Path to the output sampled COCO JSON file.
        sample_size (int): Number of categories to sample.
    """
    with open(coco_file, 'r') as f:
        coco = json.load(f)

    sampled_categories = random.sample(coco['categories'], sample_size)
    sampled_category_ids = {cat['id'] for cat in sampled_categories}
    sampled_annotations = [ann for ann in coco['annotations'] if ann['category_id'] in sampled_category_ids]
    sampled_image_ids = {ann['image_id'] for ann in sampled_annotations}
    sampled_images = [img for img in coco['images'] if img['id'] in sampled_image_ids]

    sampled_coco = {
        'images': sampled_images,
        'annotations': sampled_annotations,
        'categories': sampled_categories
    }

    with open(output_file, "w") as f:
        json.dump(sampled_coco, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample categories with related images and annotations from a COCO JSON file.")
    parser.add_argument("coco_file", type=str, help="Path to the COCO JSON file")
    parser.add_argument("output_file", type=str, help="Path to the output sampled COCO JSON file")
    parser.add_argument("sample_size", type=int, help="Number of categories to sample")
    args = parser.parse_args()

    sample_categories(args.coco_file, args.output_file, args.sample_size)