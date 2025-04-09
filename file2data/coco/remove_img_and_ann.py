"""
remove image with specifix prefix
"""
import os
import os.path as osp
from tqdm import tqdm
import argparse
from file2data import load_json, save_json
from file2data.utils import parallelise
from functools import partial


def check_img(img_info: dict, root_dirs: tuple[str]) -> tuple[bool, dict]:
    """
    check if image exists in root_dirs
    if not, set img_info['file_name'] to abs path
    """
    file_name = img_info["file_name"]
    if file_name.startswith(root_dirs):
        return False, img_info

    return True, img_info
        

def clean_img_and_ann(coco_file: str, output_file: str, root_dirs: list[str]) -> None:
    coco = load_json(coco_file)
    invalid_img_ids = set()

    # check if image exists in root_dirs
    check_fun = partial(check_img, root_dirs=tuple(root_dirs))
    check_results = parallelise(
        check_fun,
        coco["images"],
        task_type="io_bound"
    )
    invalid_img_path = []
    for flag, img_info in tqdm(check_results):
        if not flag:
            invalid_img_ids.add(img_info["id"])
            invalid_img_path.append(img_info["file_name"])

    invalid_ratio = len(invalid_img_ids) / len(coco["images"])
    print(f"invalid_img_ids: {len(invalid_img_ids)}, ratio: {invalid_ratio:.2%}")
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
    if len(invalid_img_path) > 0:
        invalid_img_txt = osp.splitext(output_file)[0] + "_invalid_img.txt"
        with open(invalid_img_txt, "w") as f:
            for img_path in invalid_img_path:
                f.write(img_path + "\n")
        print(f"invalid_img_txt: {invalid_img_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--bad_root_dirs", type=str, required=False, default=[], nargs="+")
    args = parser.parse_args()
    clean_img_and_ann(args.coco_file, args.output_file, args.bad_root_dirs)
    print(f"save to {args.output_file}")
