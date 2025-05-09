"""
init fsx file system (lustre) with coco dataset

usage:
python3 file2data/aws/init_fsx.py \
    --coco_file <coco_file> \
    --origin_img_dir <origin_img_root> \
    --fsx_img_dir <fsx_img_root> \
    --output_file <output_file> \
    --num_workers <num_workers> \
    --sudo
"""

import argparse
from file2data import load_json, save_json, load_file
from file2data.utils import parallelise
import os.path as osp
import subprocess
import os

SUDO = False

def lfs_restore(img_path: str) -> dict:
    """hsm_restore img_path to fsx_img_dir"""
    if SUDO:
        cmd = f"sudo lfs hsm_restore '{img_path}'"
    else:
        cmd = f"lfs hsm_restore '{img_path}'"

    try:
        subprocess.run(cmd, shell=True, check=True)
    except Exception as e:
        print(f"error: {e}")
        return dict(success=False, img_path=img_path, error=str(e))
    return dict(success=True, img_path=img_path)

def init_fsx(coco_file: str, origin_img_dir: str, fsx_img_dir: str, output_file: str, num_workers: int) -> None:
    """初始化fsx文件系统（lustre）
    """
    if coco_file.endswith('.json'):
        coco_data = load_json(coco_file)
    elif coco_file.endswith('.txt'):
        coco_data = {}
        coco_data['images'] = []
        # support image and other file
        img_path_list = load_file(coco_file)
        for idx, img_path in enumerate(img_path_list):
            coco_data['images'].append({
                'id': idx,
                'file_name': img_path,
            })
    elif osp.isdir(coco_file):
        # recursive search all files in directory, support image and other file
        coco_data = {}
        coco_data['images'] = []
        idx = 0
        for root, dirs, files in os.walk(coco_file):
            for file in files:
                coco_data['images'].append({
                    'id': idx,
                    'file_name': osp.join(root, file),
                })
                idx += 1
    else:
        raise ValueError(f"invalid coco file: {coco_file}")
    
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
        elif not osp.isabs(img_path):
            fsx_img_path = osp.join(fsx_img_dir, img_path)
            img_info['file_name'] = fsx_img_path
            fsx_files.append(fsx_img_path)
        else:
            fail_count += 1
            if fail_count < 3:
                print(f"error: {img_path} not in {origin_img_dir}")

    abs_fail_count = fail_count
    results = parallelise(lfs_restore, fsx_files, num_workers=num_workers)
    for result in results:
        if result['success']:
            success_count += 1
        else:
            fail_count += 1
            if fail_count < abs_fail_count + 3:
                print(f"error: {result['error']}")
    print(f"success_count: {success_count}, fail_count: {fail_count}")
    print(f'success_rate: {success_count / (success_count + fail_count)}')
    if output_file:
        save_json(output_file, coco_data)
        print(f"save coco data to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_file", type=str, required=True)
    parser.add_argument("--origin_img_dir", type=str, default="/", help="origin image directory, used for absolute path")
    parser.add_argument("--fsx_img_dir", type=str, default="/", help="fsx image directory, used for relative path and absolute path")
    parser.add_argument("--output_file", type=str, default=None, help="output coco file, if not provided, the coco file will not be saved")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers")
    parser.add_argument("--sudo", action='store_true', help="use sudo to run lfs commands")
    args = parser.parse_args()

    SUDO = args.sudo
    if SUDO:
        print("use sudo to run lfs commands")
    init_fsx(args.coco_file, args.origin_img_dir, args.fsx_img_dir, args.output_file, args.num_workers)