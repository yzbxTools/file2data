"""
init fsx file system (lustre) with coco dataset

usage:
python3 file2data/aws/init_fsx.py \
    --coco_file <coco_file> \
    --origin_img_dir <origin_img_root> \
    --fsx_img_dir <fsx_img_root> \
    --output_file <output_file>
"""

import argparse
from file2data import load_json, save_json
import os.path as osp


def init_fsx(coco_file: str, origin_img_dir: str, fsx_img_dir: str, output_file: str) -> None:
    """初始化fsx文件系统（lustre）
    """
    coco_data = load_json(coco_file)
    success_count = 0
    fail_count = 0
    for img_info in coco_data['images']:
        img_path = img_info['file_name']
        if osp.isabs(img_path) and img_path.startswith(origin_img_dir):
            fsx_img_path = osp.join(fsx_img_dir, osp.relpath(img_path, origin_img_dir))
            success_count += 1
            img_info['file_name'] = fsx_img_path
        else:
            if fail_count < 3:
                print(f"img_path: {img_path} is not abs path or not start with {origin_img_dir}")
            fail_count += 1

    print(f"success_count: {success_count}, fail_count: {fail_count}")
    print(f'success_rate: {success_count / (success_count + fail_count)}')
    if output_file:
        save_json(output_file, coco_data)
        print(f"save coco data to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_file", type=str, required=True)
    parser.add_argument("--origin_img_dir", type=str, required=True, default="/")
    parser.add_argument("--fsx_img_dir", type=str, required=True, default="/")
    parser.add_argument("--output_file", type=str, required=False, default=None)
    args = parser.parse_args()
    init_fsx(args.coco_file, args.origin_img_dir, args.fsx_img_dir, args.output_file)