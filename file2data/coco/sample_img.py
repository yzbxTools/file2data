import json
import argparse
import random
from typing import Dict, List

def sample_images_annotations(coco_file: str, output_file: str, sample_size: int) -> None:
    """
    Sample images and related annotations from a COCO JSON file.

    Args:
        coco_file (str): Path to the COCO JSON file.
        output_file (str): Path to the output sampled COCO JSON file.
        sample_size (int): Number of images to sample.
    """
    with open(coco_file, 'r') as f:
        coco = json.load(f)

    sampled_images = random.sample(coco['images'], sample_size)
    sampled_image_ids = {img['id'] for img in sampled_images}
    sampled_annotations = [ann for ann in coco['annotations'] if ann['image_id'] in sampled_image_ids]

    sampled_coco = {
        'images': sampled_images,
        'annotations': sampled_annotations,
        'categories': coco['categories']
    }

    with open(output_file, 'w') as f:
        json.dump(sampled_coco, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample images and related annotations from a COCO JSON file.")
    parser.add_argument("coco_file", type=str, help="Path to the COCO JSON file")
    parser.add_argument("output_file", type=str, help="Path to the output sampled COCO JSON file")
    parser.add_argument("sample_size", type=int, help="Number of images to sample")
    args = parser.parse_args()

    sample_images_annotations(args.coco_file, args.output_file, args.sample_size)