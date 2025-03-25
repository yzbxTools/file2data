import json
import argparse
from tqdm import tqdm
import os
from typing import List, Dict


def merge_images_annotations(
    json_files: List[str], img_dirs: List[str], output_file: str
) -> None:
    """
    Merge images and annotations from multiple COCO JSON files.

    Args:
        json_files (List[str]): List of paths to COCO JSON files.
        output_file (str): Path to the output merged COCO JSON file.
    """
    merged_coco = None
    img_id_offset = 0
    ann_id_offset = 0

    for json_file, img_dir in zip(json_files, img_dirs):
        with open(json_file, "r") as f:
            coco = json.load(f)

        if img_dir:
            for img_info in tqdm(coco["images"], desc=f"Processing {json_file}"):
                img_info["file_name"] = os.path.join(img_dir, img_info["file_name"])

        if merged_coco is None:
            merged_coco = coco
        else:
            for img in coco["images"]:
                img["id"] += img_id_offset
            for ann in coco["annotations"]:
                ann["id"] += ann_id_offset
                ann["image_id"] += img_id_offset

            merged_coco["images"].extend(coco["images"])
            merged_coco["annotations"].extend(coco["annotations"])

        img_id_offset = max(img["id"] for img in merged_coco["images"]) + 1
        ann_id_offset = max(ann["id"] for ann in merged_coco["annotations"]) + 1

    with open(output_file, "w") as f:
        json.dump(merged_coco, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge images and annotations from multiple COCO JSON files."
    )
    parser.add_argument(
        "--json_files", type=str, nargs="+", help="List of paths to COCO JSON files"
    )
    parser.add_argument(
        "--img_dirs",
        type=str,
        nargs="+",
        help="List of paths to image directories",
        default=None,
    )
    parser.add_argument(
        "--output_file", type=str, help="Path to the output merged COCO JSON file"
    )
    args = parser.parse_args()

    if args.img_dirs is None:
        args.img_dirs = [""] * len(args.json_files)
    elif len(args.img_dirs) != len(args.json_files):
        raise ValueError(
            "The number of image directories must be equal to the number of JSON files."
        )
    merge_images_annotations(args.json_files, args.img_dirs, args.output_file)
    print(f"Merged COCO JSON file saved to {args.output_file}")