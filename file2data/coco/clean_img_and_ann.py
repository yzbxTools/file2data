"""
remove invalid images and annotations from coco dataset

invalid images: osp.exists(img_path) == False
invalid annotations: ann['image_id'] not in invalid_img_ids
"""

import os
import os.path as osp
from tqdm import tqdm
import argparse
from file2data import load_json, save_json


def clean_img_and_ann(coco_file: str, output_file: str, root_dirs: list[str]) -> None:
    coco = load_json(coco_file)
    invalid_img_ids = set()
    rel_img_ids = set()
    for img_info in tqdm(coco["images"]):
        file_name = img_info["file_name"]
        if osp.isabs(file_name):
            if not osp.exists(file_name):
                invalid_img_ids.add(img_info["id"])
        else:
            exist = False
            for root_dir in root_dirs:
                if osp.exists(osp.join(root_dir, file_name)):
                    img_info["file_name"] = osp.join(root_dir, file_name)
                    exist = True
                    break
            if not exist:
                invalid_img_ids.add(img_info["id"])
                rel_img_ids.add(img_info["id"])
    print(f"invalid_img_ids: {len(invalid_img_ids)}")
    print(f"rel_img_ids: {len(rel_img_ids)}")
    clean_coco = {
        "images": [
            img_info
            for img_info in coco["images"]
            if img_info["id"] not in invalid_img_ids
        ],
        "annotations": [
            ann for ann in coco["annotations"] if ann["image_id"] not in invalid_img_ids
        ],
        "categories": coco["categories"],
    }
    os.makedirs(osp.dirname(output_file), exist_ok=True)
    save_json(output_file, clean_coco)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--root_dirs", type=str, required=False, default=[], nargs="+")
    args = parser.parse_args()
    clean_img_and_ann(args.coco_file, args.output_file, args.root_dirs)
    print(f"save to {args.output_file}")
