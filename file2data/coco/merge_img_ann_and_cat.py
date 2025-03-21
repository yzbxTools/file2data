from file2data import load_json, save_json
import argparse
import os
import sys
from typing import List, Dict


def merge_all(json_files: List[str], img_dirs: List[str], output_file: str) -> None:
    """
    Merge images, annotations, and categories from multiple COCO JSON files.
    Update image paths to absolute paths using the provided image directories.

    Args:
        json_files (List[str]): List of paths to COCO JSON files.
        img_dirs (List[str]): List of image directories corresponding to each JSON file.
        output_file (str): Path to the output merged COCO JSON file.
    """
    if not json_files:
        print("错误: 未提供JSON文件列表")
        return
    
    if not img_dirs:
        print("错误: 未提供图片目录列表")
        return
    
    merged_coco = None
    img_id_offset = 0
    ann_id_offset = 0
    cat_id_offset = 0

    for i, json_file in enumerate(json_files):
        if not os.path.exists(json_file):
            print(f"警告: JSON文件不存在: {json_file}，已跳过")
            continue
            
        img_dir = img_dirs[i] if i < len(img_dirs) else img_dirs[-1]
        
        if not os.path.exists(img_dir):
            print(f"警告: 图片目录不存在: {img_dir}，将使用相对路径")
        
        try:
            coco = load_json(json_file)
            
            # 验证COCO格式
            required_keys = ["images", "annotations", "categories"]
            if not all(key in coco for key in required_keys):
                print(f"警告: {json_file} 不是有效的COCO格式文件，缺少必要字段")
                continue
                
            # 更新图片路径为绝对路径
            if img_dir and os.path.exists(img_dir):
                for img in coco["images"]:
                    # 替换相对路径为绝对路径
                    img["file_name"] = os.path.abspath(
                        os.path.join(img_dir, img["file_name"])
                    )

            if merged_coco is None:
                merged_coco = coco
            else:
                for img in coco["images"]:
                    img["id"] += img_id_offset
                for ann in coco["annotations"]:
                    ann["id"] += ann_id_offset
                    ann["image_id"] += img_id_offset
                    ann["category_id"] += cat_id_offset
                for cat in coco["categories"]:
                    cat["id"] += cat_id_offset

                merged_coco["images"].extend(coco["images"])
                merged_coco["annotations"].extend(coco["annotations"])
                merged_coco["categories"].extend(coco["categories"])

            if merged_coco["images"]:
                img_id_offset = max(img["id"] for img in merged_coco["images"]) + 1
            if merged_coco["annotations"]:
                ann_id_offset = max(ann["id"] for ann in merged_coco["annotations"]) + 1
            if merged_coco["categories"]:
                cat_id_offset = max(cat["id"] for cat in merged_coco["categories"]) + 1
        
        except Exception as e:
            print(f"处理文件 {json_file} 时出错: {str(e)}")
            continue

    if merged_coco is None:
        print("错误: 没有成功处理任何COCO文件，无法生成合并结果")
        return
        
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            print(f"创建输出目录时出错: {str(e)}")
            return
    
    try:
        save_json(output_file, merged_coco)
        print(f"成功合并 {len(json_files)} 个COCO文件，结果保存至: {output_file}")
        print(f"合并结果包含 {len(merged_coco['images'])} 个图像, {len(merged_coco['annotations'])} 个标注, {len(merged_coco['categories'])} 个类别")
    except Exception as e:
        print(f"保存结果时出错: {str(e)}")


def validate_args(args):
    """验证命令行参数"""
    if not args.json_files:
        print("错误: 必须提供至少一个JSON文件")
        return False
        
    if not args.img_dirs:
        print("错误: 必须提供至少一个图片目录")
        return False
        
    if not args.output_file:
        print("错误: 必须提供输出文件路径")
        return False
        
    # 验证所有JSON文件是否存在
    missing_files = [f for f in args.json_files if not os.path.exists(f)]
    if missing_files:
        print(f"错误: 以下JSON文件不存在: {', '.join(missing_files)}")
        return False
        
    # 验证所有图片目录是否存在
    missing_dirs = [d for d in args.img_dirs if not os.path.exists(d)]
    if missing_dirs:
        print(f"警告: 以下图片目录不存在: {', '.join(missing_dirs)}")
        
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="合并多个COCO JSON文件的图像、标注和类别，并使用绝对图片路径。"
    )
    parser.add_argument(
        "--json_files", type=str, nargs="+", help="COCO JSON文件路径列表"
    )
    parser.add_argument(
        "--img_dirs",
        type=str,
        nargs="+",
        help="对应每个JSON文件的图片目录列表"
    )
    parser.add_argument(
        "--output_file", type=str, help="输出合并后的COCO JSON文件路径"
    )
    args = parser.parse_args()

    # 验证参数
    if not validate_args(args):
        sys.exit(1)

    # 确保img_dirs的长度与json_files匹配
    if len(args.img_dirs) != len(args.json_files):
        print(
            f"警告: 图片目录数量 ({len(args.img_dirs)}) 与JSON文件数量 ({len(args.json_files)}) 不匹配"
        )
        print("将使用最后一个图片目录作为默认目录")

    # 执行合并
    merge_all(args.json_files, args.img_dirs, args.output_file)
