from file2data import load_json, save_json
import argparse
import os
from typing import List, Dict


def merge_all(json_files: List[str], img_dirs: List[str], output_file: str) -> None:
    """
    Merge images, annotations, and categories from multiple COCO JSON files.
    Update image paths to absolute paths using the provided image directories.

    Args:
        json_files (List[str]): List of paths to COCO JSON files.
        img_dirs (List[str]): List of image directories corresponding to each JSON file.
        output_file (str): Path to the output merged COCO JSON file.
    """
    merged_coco = None
    img_id_offset = 0
    ann_id_offset = 0
    cat_id_offset = 0

    for i, json_file in enumerate(json_files):
        img_dir = img_dirs[i] if i < len(img_dirs) else img_dirs[-1]

        coco = load_json(json_file)

        # Update image file paths to absolute paths if img_dir is provided
        if img_dir:
            for img in coco["images"]:
                # Replace relative path with absolute path
                img["file_name"] = os.path.abspath(
                    os.path.join(img_dir, img["file_name"])
                )

        if merged_coco is None:
            merged_coco = coco
        else:
            for img in coco["images"]:
                img["id"] += img_id_offset
            for ann in coco["annotations"]:
                ann["id"] += ann_id_offset
                ann["image_id"] += img_id_offset
                ann["category_id"] += cat_id_offset
            for cat in coco["categories"]:
                cat["id"] += cat_id_offset

            merged_coco["images"].extend(coco["images"])
            merged_coco["annotations"].extend(coco["annotations"])
            merged_coco["categories"].extend(coco["categories"])

        img_id_offset = max(img["id"] for img in merged_coco["images"]) + 1
        ann_id_offset = max(ann["id"] for ann in merged_coco["annotations"]) + 1
        cat_id_offset = max(cat["id"] for cat in merged_coco["categories"]) + 1

    save_json(merged_coco, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge images, annotations, and categories from multiple COCO JSON files."
    )
    parser.add_argument(
        "--json_files", type=str, nargs="+", help="List of paths to COCO JSON files"
    )
    parser.add_argument(
        "--img_dirs",
        type=str,
        nargs="+",
        help="List of image directories corresponding to each JSON file",
    )
    parser.add_argument(
        "--output_file", type=str, help="Path to the output merged COCO JSON file"
    )
    args = parser.parse_args()

    # Ensure img_dirs is provided and has the same length as json_files
    if args.img_dirs and len(args.img_dirs) != len(args.json_files):
        print(
            f"Warning: Number of image directories ({len(args.img_dirs)}) doesn't match number of JSON files ({len(args.json_files)})"
        )

    merge_all(args.json_files, args.img_dirs, args.output_file)
