"""
remove invalid images and annotations from coco dataset

invalid images: osp.exists(img_path) == False
invalid annotations: ann['image_id'] not in invalid_img_ids
"""

import os
import os.path as osp
from tqdm import tqdm
from PIL import Image
import argparse
from file2data import load_json, save_json
from file2data.utils import parallelise
from functools import partial
from thefuzz import fuzz, process


def verify_image(img_path: str, verbose: bool = False) -> bool:
    if osp.exists(img_path):
        if osp.getsize(img_path) < 1024:
            if verbose:
                print(f"Image file size too small: {img_path}")
            return False

        try:
            # verify images
            im = Image.open(img_path)
            im.verify()  # PIL verify
            shape = im.size
            assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
            if im.format.lower() in ("jpg", "jpeg"):
                with open(img_path, "rb") as f:
                    f.seek(-2, 2)
                    if f.read() != b"\xff\xd9":  # corrupt JPEG
                        if verbose:
                            print(f"find corrupt JPEG: {img_path}")
                        return False
        except Exception as e:
            if verbose:
                print(f"Error in load image: {img_path}, {e}")
            return False
        return True
    else:
        if verbose:
            print(f"Image not found: {img_path}")
        return False


def check_img(img_info: dict, root_dirs: list[str]) -> tuple[bool, dict]:
    """
    check if image exists in root_dirs
    if not, set img_info['file_name'] to abs path
    """
    file_name = img_info["file_name"]
    if osp.isabs(file_name):
        if not osp.exists(file_name):
            return False, img_info
    else:
        exist = False
        for root_dir in root_dirs:
            if osp.exists(osp.join(root_dir, file_name)):
                img_info["file_name"] = osp.join(root_dir, file_name)
                exist = True
                break
        if not exist:
            return False, img_info

    # check if image_path is a valid image
    file_name = img_info["file_name"]
    is_valid = verify_image(file_name)
    return is_valid, img_info


def clean_img_and_ann(coco_file: str, output_file: str, img_database: dict, img_database_dir: str, root_dirs: list[str], chunksize: int = 100) -> None:
    """
    clean invalid images and annotations from coco dataset
    coco_file: path to coco dataset
    output_file: path to output coco dataset
    img_database:
        - {file_name: {md5_value: file_path}}   # new format from img_txt.py
        - {file_name: {maps: {md5_value: file_path}}}  # old format from img_txt.py
    img_database_dir: root image directory for path in img_database_file
    root_dirs: list of root directories
    chunksize: number of images to process in parallel
    """
    coco = load_json(coco_file)
    invalid_img_ids = set()

    # check if image exists in root_dirs
    check_fun = partial(check_img, root_dirs=root_dirs)
    check_results = parallelise(check_fun, coco["images"], chunksize=chunksize, task_type="io_bound")
    invalid_img_path = []

    replace_img_number = 0
    reject_img_number = 0
    # file_name: matched_file_name
    replace_img_map = {}
    reject_img_map = {}
    for flag, img_info in tqdm(check_results):
        if not flag:
            base_name = osp.basename(img_info["file_name"])
            if base_name in img_database:
                if 'maps' in img_database[base_name]:
                    file_choices = list(img_database[base_name]['maps'].values())
                else:
                    file_choices = list(img_database[base_name].values())

                # threshold = base_name
                file_query = img_info["file_name"]
                base_name_ratio = fuzz.ratio(file_query, base_name)
                best_match, best_match_ratio = process.extractOne(file_query, file_choices, scorer=fuzz.ratio)
                if best_match_ratio > base_name_ratio:
                    replace_img_map[img_info["file_name"]] = best_match
                    replace_img_number += 1
                    if replace_img_number < 3:
                        print(f"replace {img_info['file_name']} with {best_match}")
                    img_info["file_name"] = osp.join(img_database_dir, best_match)
                else:
                    reject_img_map[img_info["file_name"]] = best_match
                    reject_img_number += 1
                    if reject_img_number < 3:
                        print(f"reject match {best_match_ratio} for {img_info['file_name']}")
                    invalid_img_ids.add(img_info["id"])
                    invalid_img_path.append(img_info["file_name"])
            else:
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
        print(f"invalid_img_txt: {invalid_img_txt}, num: {len(invalid_img_path)}")

    if len(replace_img_map) > 0:
        replace_img_txt = osp.splitext(output_file)[0] + "_replace_img.txt"
        with open(replace_img_txt, "w") as f:
            for img_path, matched_img_path in replace_img_map.items():
                f.write(f"{img_path} {matched_img_path}\n")
        print(f"replace_img_txt: {replace_img_txt}, num: {len(replace_img_map)}")
    
    if len(reject_img_map) > 0:
        reject_img_txt = osp.splitext(output_file)[0] + "_reject_img.txt"
        with open(reject_img_txt, "w") as f:
            for img_path, matched_img_path in reject_img_map.items():
                f.write(f"{img_path} {matched_img_path}\n")
        print(f"reject_img_txt: {reject_img_txt}, num: {len(reject_img_map)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--img_database_file", type=str, required=False, default="", help="path to img database")
    parser.add_argument("--img_database_dir", type=str, required=False, default="", help="root image directory for path in img_database_file")
    parser.add_argument("--root_dirs", type=str, required=False, default=[], nargs="+")
    parser.add_argument("--chunksize", type=int, required=False, default=16)
    args = parser.parse_args()
    if args.img_database_file:
        database = load_json(args.img_database_file)
    else:
        database = {}

    clean_img_and_ann(args.coco_file, args.output_file, database, args.img_database_dir, args.root_dirs, args.chunksize)
    print(f"save to {args.output_file}")
