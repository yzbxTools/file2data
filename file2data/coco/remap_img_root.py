"""
remap image root directory in coco dataset

usage:
python remap_img_root.py \
    --coco_file <coco_file> \
    --old_roots <old_root1> <old_root2> \
    --new_root <new_root1> <new_root2>  \
    --output_file <output_file>
"""

import argparse
import os
import os.path as osp
from tqdm import tqdm
from file2data import load_json, save_json


def remap_img_root(coco_file: str, old_roots: list[str], new_roots: list[str], output_file: str) -> None:
    """重映射COCO数据集中图片的根目录路径
    
    Args:
        coco_file: COCO标注文件路径
        old_roots: 原始根目录路径列表
        new_roots: 新的根目录路径列表
        output_file: 输出文件路径
    """
    coco = load_json(coco_file)
    
    rel_img_count = 0
    root_map_test = set()
    for img_info in tqdm(coco["images"], desc="remap image root"):
        file_name = img_info["file_name"]
        # 处理绝对路径
        if osp.isabs(file_name):
            for old_root, new_root in zip(old_roots, new_roots):
                if file_name.startswith(old_root):
                    rel_path = osp.relpath(file_name, old_root)
                    img_info["file_name"] = osp.join(new_root, rel_path)

                    root_map = f"{old_root} -> {new_root}"
                    if root_map not in root_map_test:
                        root_map_test.add(root_map)
                        if not osp.exists(img_info["file_name"]):
                            print(f"remap image root {root_map} failed: {img_info['file_name']} not exists")
                    break
        # 处理相对路径
        else:
            rel_img_count += 1
            if rel_img_count < 3:
                print(f"rel_img_count: {rel_img_count}, file_name: {file_name}")
    os.makedirs(osp.dirname(output_file), exist_ok=True)
    save_json(output_file, coco)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_file", type=str, required=True, help="COCO标注文件路径")
    parser.add_argument("--old_roots", type=str, nargs="+", required=True, help="原始根目录路径列表")
    parser.add_argument("--new_roots", type=str, nargs="+", required=True, help="新的根目录路径列表")
    parser.add_argument("--output_file", type=str, required=True, help="输出文件路径")
    args = parser.parse_args()
    
    remap_img_root(args.coco_file, args.old_roots, args.new_roots, args.output_file)
    print(f"保存到 {args.output_file}")
