"""
image txt processing: build file_name2path map and check md5sum for deduplication

args:
    - image txt file: image txt, each line in a relative image file_path
    - output file: image info list output json file
    - image dir

image info:
    - file_name: os.path.basename(file_path)
    - maps:
        - md5sum1: file_path1
        - md5sum2: file_path2
        - ...
"""

import argparse
import os
import hashlib
from tqdm import tqdm
from file2data import load_json, save_json
from file2data.utils import parallelise
from file2data.coco.clean_img_and_ann import verify_image
from typing import Dict, List, Tuple, Any


def calculate_md5(file_path: str) -> str:
    """
    计算文件的MD5值

    Args:
        file_path: 文件路径

    Returns:
        md5值
    """
    if not os.path.exists(file_path):
        return ""

    md5_hash = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            # 分块读取文件以处理大文件
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    except Exception as e:
        print(f"计算MD5值时出错 {file_path}: {e}")
        return ""


def process_image_file(file_path: str, img_dir: str) -> Dict[str, Any]:
    """
    处理单个图像文件，计算其MD5值

    Args:
        file_path: 相对文件路径
        img_dir: 图像目录

    Returns:
        包含文件名和MD5映射的字典
    """
    abs_path = os.path.join(img_dir, file_path)
    file_name = os.path.basename(file_path)
    if not verify_image(abs_path):
        return {"file_name": file_name, "maps": {}, "is_valid": False}
    md5_value = calculate_md5(abs_path)

    return {"file_name": file_name, "maps": {md5_value: file_path}, "is_valid": True}


def process_image_txt(
    txt_file: str, img_dir: str, output_file: str, num_workers: int = 0
) -> None:
    """
    处理图像文本文件，构建文件名到路径的映射，并检查MD5值以进行去重

    Args:
        txt_file: 图像文本文件路径
        img_dir: 图像目录
        output_file: 输出JSON文件路径
        num_workers: 并行处理的工作线程数
    """
    # 读取图像文本文件
    with open(txt_file, "r") as f:
        image_paths = [line.strip() for line in f.readlines()]

    print(f"读取了 {len(image_paths)} 个图像路径")

    # 并行处理图像文件
    func = lambda x: process_image_file(x, img_dir)
    results = parallelise(func, image_paths, num_workers=num_workers, verbose=True, task_type="io_bound")

    # 合并相同文件名的结果
    file_name_to_info: Dict[str, Dict[str, Any]] = {}
    for result in results:
        if not result["is_valid"]:
            continue
        file_name = result["file_name"]
        if file_name in file_name_to_info:
            # 合并maps
            file_name_to_info[file_name].update(result["maps"])
        else:
            file_name_to_info[file_name] = result["maps"]

    # 保存结果
    save_json(output_file, file_name_to_info)
    print(f"处理完成，共 {len(file_name_to_info)} 个唯一文件名")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="处理图像文本文件，构建文件名到路径的映射，并检查MD5值以进行去重"
    )
    parser.add_argument("--txt_file", type=str, required=True, help="图像文本文件路径")
    parser.add_argument("--img_dir", type=str, required=True, help="图像目录")
    parser.add_argument(
        "--output_file", type=str, required=True, help="输出JSON文件路径"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="并行处理的工作线程数"
    )
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    process_image_txt(args.txt_file, args.img_dir, args.output_file, args.num_workers)


if __name__ == "__main__":
    main()
