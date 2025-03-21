"""
modify relative path to absolute path in coco dataset
"""

import os
import os.path as osp
from tqdm import tqdm

from file2data import load_json, save_json
import argparse


def modify_rel2abs(coco_file: str, output_file: str, root_dirs: list[str]) -> None:
    coco = load_json(coco_file)
    origin_abs_invalid_count = 0
    origin_abs_valid_count = 0
    rel2abs_valid_count = 0
    rel2abs_invalid_count = 0
    for img_info in tqdm(coco["images"]):
        file_name = img_info["file_name"]
        if osp.isabs(file_name):
            if osp.exists(file_name):
                origin_abs_valid_count += 1
            else:
                origin_abs_invalid_count += 1
            continue

        exist = False
        for root_dir in root_dirs:
            if osp.exists(osp.join(root_dir, file_name)):
                img_info["file_name"] = osp.join(root_dir, file_name)
                exist = True
                break
        if exist:
            rel2abs_valid_count += 1
        else:
            rel2abs_invalid_count += 1
    print(f"origin_abs_valid_count: {origin_abs_valid_count}")
    print(f"origin_abs_invalid_count: {origin_abs_invalid_count}")
    print(f"rel2abs_valid_count: {rel2abs_valid_count}")
    print(f"rel2abs_invalid_count: {rel2abs_invalid_count}")
    os.makedirs(osp.dirname(output_file), exist_ok=True)
    save_json(output_file, coco)
    print(f"save to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=False)
    parser.add_argument("--root_dirs", type=str, required=False, default=[], nargs="+")
    args = parser.parse_args()

    if not args.output_file:
        args.output_file = args.coco_file.replace(".json", "_abs.json")

    modify_rel2abs(args.coco_file, args.output_file, args.root_dirs)
