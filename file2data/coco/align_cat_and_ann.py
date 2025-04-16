"""
align categories and annotations by category names

usage:
python3 file2data/coco/align_cat_and_ann.py \
    --ref_coco_file <ref_coco_file> \
    --input_coco_file <input_coco_file> \
    --output_file <output_file>
"""

import argparse
from tqdm import tqdm
from file2data import load_json, save_json
from loguru import logger


def align_cat_and_ann(
    ref_coco_file: str, input_coco_file: str, output_file: str
) -> None:
    """
    align categories and annotations
    """
    ref_coco = load_json(ref_coco_file)
    input_coco = load_json(input_coco_file)

    input_id2cat_name = {cat["id"]: cat["name"] for cat in input_coco["categories"]}
    cat_name2ref_id = {cat["name"]: cat["id"] for cat in ref_coco["categories"]}

    input_id2ref_id = {}
    for input_id, input_cat_name in input_id2cat_name.items():
        if input_cat_name not in cat_name2ref_id:
            logger.warning(
                f"category {input_cat_name} not found in reference categories, "
                f"skipping category {input_id}"
            )
            ref_id = -1
        else:
            ref_id = cat_name2ref_id[input_cat_name]
        input_id2ref_id[input_id] = ref_id

    for ann in tqdm(input_coco["annotations"]):
        input_id = ann["category_id"]
        ref_id = input_id2ref_id[input_id]
        ann["category_id"] = ref_id

    input_coco["categories"] = ref_coco["categories"]
    save_json(output_file, input_coco)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_coco_file", type=str, required=True)
    parser.add_argument("--input_coco_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    align_cat_and_ann(args.ref_coco_file, args.input_coco_file, args.output_file)
