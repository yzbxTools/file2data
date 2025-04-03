"""
init fsx file system (lustre) with coco dataset

usage:
python3 file2data/aws/init_fsx.py \
    --coco_file <coco_file> \
    --origin_img_dir <origin_img_root> \
    --fsx_img_dir <fsx_img_root> \
    --output_file <output_file> \
    --num_workers <num_workers>
"""

import argparse
from file2data import load_json, save_json
from file2data.utils import parallelise
import os.path as osp
import subprocess
import os


def lfs_restore(img_path: str) -> bool:
    """hsm_restore img_path to fsx_img_dir"""
    cmd = f"lfs hsm_restore {img_path}"
    try:
        subprocess.run(cmd, shell=True, check=True)
    except Exception as e:
        print(f"error: {e}")
        return False
    return True

def init_fsx(coco_file: str, origin_img_dir: str, fsx_img_dir: str, output_file: str, num_workers: int) -> None:
    """初始化fsx文件系统（lustre）
    """
    coco_data = load_json(coco_file)
    success_count = 0
    fail_count = 0

    fsx_files = []
    for img_info in coco_data['images']:
        img_path = img_info['file_name']
        if osp.isabs(img_path) and img_path.startswith(origin_img_dir):
            fsx_img_path = osp.join(fsx_img_dir, osp.relpath(img_path, origin_img_dir))
            success_count += 1
            img_info['file_name'] = fsx_img_path
            fsx_files.append(fsx_img_path)
        else:
            if fail_count < 3:
                print(f"img_path: {img_path} is not abs path or not start with {origin_img_dir}")
            fail_count += 1

    parallelise(lfs_restore, fsx_files, num_workers=num_workers)
    print(f"success_count: {success_count}, fail_count: {fail_count}")
    print(f'success_rate: {success_count / (success_count + fail_count)}')
    if output_file:
        save_json(output_file, coco_data)
        print(f"save coco data to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_file", type=str, required=True)
    parser.add_argument("--origin_img_dir", type=str, default="/")
    parser.add_argument("--fsx_img_dir", type=str, default="/")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    init_fsx(args.coco_file, args.origin_img_dir, args.fsx_img_dir, args.output_file, args.num_workers)