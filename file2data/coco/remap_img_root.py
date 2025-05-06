"""
remap image root directory in coco dataset

usage:
python3 file2data/coco/remap_img_root.py \
    --coco_file <coco_file> \
    --old_roots <old_root1> <old_root2> \
    --new_root <new_root1> <new_root2>  \
    --rel2abs_dirs <dir1> <dir2> \
    --abs2rel_dirs <dir1> <dir2> \
    --output_file <output_file>

# {'/team/drive-1': 18000, '/mnt/lakedata_azure': 1340, '/data_tmp/cloth_wire_data': 23, '/ssd/data_zx': 20637}
/team/drive-1 -> /team/drive-1
/mnt/lakedata_azure -> /team/drive-1/ssd/mnt/lakedata_azure
/data_tmp/cloth_wire_data -> /team/drive-1/data_ap/train/all_images
/ssd/data_zx -> /team/drive-1/ssd/data_zx

python3 file2data/coco/remap_img_root.py \
    --coco_file /mnt/fsx/workspace_robin/datasets/ap59_zx5_train_sampled_40000.json \
    --old_roots /mnt/lakedata_azure /data_tmp/cloth_wire_data /ssd/data_zx \
    --new_roots /team/drive-1/ssd/mnt/lakedata_azure /team/drive-1/data_ap/train/all_images /team/drive-1/ssd/data_zx \
    --output_file /mnt/fsx/workspace_robin/datasets/ap59_zx5_train_sampled_40000_remap.json
"""

import argparse
import os
import os.path as osp
from tqdm import tqdm
from file2data import load_json, save_json


def remap_img_root(
    coco_file: str,
    old_roots: list[str],
    new_roots: list[str],
    rel2abs_dirs: list[str],
    abs2rel_dirs: list[str],
    output_file: str,
) -> None:
    """重映射COCO数据集中图片的根目录路径

    Args:
        coco_file: COCO标注文件路径
        old_roots: 原始根目录路径列表
        new_roots: 新的根目录路径列表
        rel2abs_dirs: 相对路径列表，将相对路径转换为绝对路径
        abs2rel_dirs: 相对路径列表，将绝对路径统一转换为相对路径，便于更换根目录
        output_file: 输出文件路径
    """
    coco = load_json(coco_file)

    rel_img_count = 0
    abs2rel_count = 0
    root_map_test = set()
    for img_info in tqdm(coco["images"], desc="remap image root"):
        file_name = img_info["file_name"]

        # convert relative path to absolute path
        if not osp.isabs(file_name):
            for img_dir in rel2abs_dirs:
                new_file_name = osp.join(img_dir, file_name)
                if osp.exists(new_file_name):
                    img_info["file_name"] = new_file_name
                    break

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
                            print(
                                f"remap image root {root_map} failed: {img_info['file_name']} not exists"
                            )
                    break
            
            # 将绝对路径统一转换为相对路径，便于更换根目录
            if abs2rel_dirs:
                file_name = img_info["file_name"]
                for rel_dir in abs2rel_dirs:
                    if file_name.startswith(rel_dir):
                        abs2rel_count += 1
                        img_info["file_name"] = osp.relpath(file_name, rel_dir)
                        break

        # 处理找不到的相对路径
        else:
            rel_img_count += 1
            if rel_img_count < 3:
                print(f"rel_img_count: {rel_img_count}, file_name: {file_name}")

    total_img_count = len(coco["images"])
    if rel_img_count > 0:
        print(f"final rel_img_count: {rel_img_count}, ratio: {rel_img_count / total_img_count:.2%}")

    if abs2rel_count:
        print(f"final abs2rel_count: {abs2rel_count}, ratio: {abs2rel_count / total_img_count:.2%}")

    if osp.dirname(output_file):
        os.makedirs(osp.dirname(output_file), exist_ok=True)
    save_json(output_file, coco)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_file", type=str, required=True, help="COCO标注文件路径")
    parser.add_argument(
        "--old_roots", type=str, nargs="+", required=True, help="原始根目录路径列表"
    )
    parser.add_argument(
        "--new_roots", type=str, nargs="+", required=True, help="新的根目录路径列表"
    )
    parser.add_argument(
        "--rel2abs_dirs",
        type=str,
        nargs="+",
        required=False,
        default=[],
        help="将相对路径转换为绝对路径的图片根目录列表",
    )
    parser.add_argument("--abs2rel_dirs", type=str, nargs="+", required=False, default=[], help="绝对路径列表，将绝对路径统一转换为相对路径，便于更换根目录")
    parser.add_argument("--output_file", type=str, required=True, help="输出文件路径")
    args = parser.parse_args()

    remap_img_root(
        args.coco_file, args.old_roots, args.new_roots, args.rel2abs_dirs, args.abs2rel_dirs, args.output_file
    )
    print(f"保存到 {args.output_file}")
