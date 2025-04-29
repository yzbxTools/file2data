import json
import argparse
from typing import List

def sample_categories(coco_file: str, output_file: str, sampled_categories: List[str]) -> None:
    """
    Sample categories with related images and annotations from a COCO JSON file.

    Args:
        coco_file (str): Path to the COCO JSON file.
        output_file (str): Path to the output sampled COCO JSON file.
        sampled_categories (List[str]): List of categories to sample.
    """
    with open(coco_file, 'r') as f:
        coco = json.load(f)

    # sampled_categories = random.sample(coco['categories'], sample_size)
    sampled_category_ids = {cat['id'] for cat in coco['categories'] if cat['name'] in sampled_categories}
    sampled_annotations = [ann for ann in coco['annotations'] if ann['category_id'] in sampled_category_ids]
    sampled_image_ids = {ann['image_id'] for ann in sampled_annotations}
    sampled_images = [img for img in coco['images'] if img['id'] in sampled_image_ids]

    sampled_coco = {
        'images': sampled_images,
        'annotations': sampled_annotations,
        'categories': [cat for cat in coco['categories'] if cat['id'] in sampled_category_ids]
    }

    with open(output_file, "w") as f:
        json.dump(sampled_coco, f, indent=2)

    origin_img_num = len(coco['images'])
    sampled_img_num = len(sampled_images)
    print(f"origin_img_num: {origin_img_num}, sampled_img_num: {sampled_img_num}")
    origin_ann_num = len(coco['annotations'])
    sampled_ann_num = len(sampled_annotations)
    print(f"origin_ann_num: {origin_ann_num}, sampled_ann_num: {sampled_ann_num}")
    print(f"sampled_img_num / origin_img_num: {sampled_img_num / origin_img_num}")
    print(f"sampled_ann_num / origin_ann_num: {sampled_ann_num / origin_ann_num}")
    print(f"save to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample categories with related images and annotations from a COCO JSON file.")
    parser.add_argument("--coco_file", type=str, help="Path to the COCO JSON file")
    parser.add_argument("--output_file", type=str, help="Path to the output sampled COCO JSON file")
    parser.add_argument("--sampled_categories", type=str, nargs="+", help="Path to the sampled categories file")
    args = parser.parse_args()

    if args.sampled_categories[0].endswith(".txt") and len(args.sampled_categories) == 1:
        with open(args.sampled_categories[0], 'r') as f:
            args.sampled_categories = [line.strip() for line in f.readlines()]
    sample_categories(args.coco_file, args.output_file, args.sampled_categories)