"""
modify relative path to absolute path in coco dataset
"""

import os
import os.path as osp

from file2data import load_json, save_json
import argparse
from file2data.utils import parallelise
from functools import partial

def rel2abs_worker(img_info: dict, root_dirs: list[str]) -> tuple[dict, str]:
    file_name = img_info["file_name"]
    if osp.isabs(file_name):
        if osp.exists(file_name):
            return img_info, "abs_valid"
        else:
            return img_info, "abs_invalid"
    else:
        for root_dir in root_dirs:
            if osp.exists(osp.join(root_dir, file_name)):
                img_info['file_name'] = osp.join(root_dir, file_name)
                return img_info, "rel2abs_valid"
    return img_info, "rel2abs_invalid"

def modify_rel2abs(coco_file: str, output_file: str, root_dirs: list[str]) -> None:
    coco = load_json(coco_file)
    counts = {"abs_valid": 0, "abs_invalid": 0, "rel2abs_valid": 0, "rel2abs_invalid": 0}
    func = partial(rel2abs_worker, root_dirs=root_dirs)
    results = parallelise(func, coco["images"], num_workers=0, verbose=True, task_type="io_bound")
    for result in results:
        counts[result[1]] += 1
    coco["images"] = [result[0] for result in results]
    print(counts)
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
