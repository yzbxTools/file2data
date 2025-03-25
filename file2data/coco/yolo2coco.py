"""
convert yolo format dataset to coco format dataset

yolo format:

root_dir/
    images/
        img1.jpg
        img2.jpg
        ...
    labels/
        img1.txt

coco format:
annotations.json
"""

import os
import json
import argparse
from PIL import Image
from tqdm import tqdm
import numpy as np
from pathlib import Path
import concurrent.futures
import threading
import imagesize  # 导入imagesize库进行快速图像尺寸读取
import datetime


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="将YOLO格式数据集转换为COCO格式")
    parser.add_argument("--root_dir", type=str, required=True, help="YOLO数据集根目录")
    parser.add_argument(
        "--output", type=str, default="annotations.json", help="输出的COCO格式注释文件"
    )
    parser.add_argument("--img_dir", type=str, default="images", help="图像目录名称")
    parser.add_argument("--label_dir", type=str, default="labels", help="标签目录名称")
    parser.add_argument(
        "--class_names",
        type=str,
        default=None,
        help="类别名称文件(每行一个类别)",
        required=True,
    )
    parser.add_argument("--num_workers", type=int, default=max(1, os.cpu_count()//2), help="并发处理的线程数")
    parser.add_argument("--recursive", action="store_true", help="是否递归遍历子目录")
    return parser.parse_args()


def get_image_files(directory, recursive=False):
    """递归获取目录中的所有图像文件"""
    image_files = []
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp")

    if recursive:
        # 递归遍历目录
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(image_extensions):
                    # 存储相对路径
                    rel_path = os.path.relpath(os.path.join(root, file), directory)
                    image_files.append(rel_path)
    else:
        # 只遍历顶层目录
        for file in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, file)) and file.lower().endswith(
                image_extensions
            ):
                image_files.append(file)

    return image_files


def yolo_bbox_to_coco(box, img_width, img_height):
    """
    将YOLO格式的边界框转换为COCO格式

    YOLO: [x_center, y_center, width, height] - 所有值都是相对于图像宽高的比例
    COCO: [x_min, y_min, width, height] - 实际像素值
    """
    x_center, y_center, width, height = box

    # 转换为像素值
    x_center = x_center * img_width
    y_center = y_center * img_height
    width = width * img_width
    height = height * img_height

    # 转换为左上角坐标
    x_min = x_center - width / 2
    y_min = y_center - height / 2

    # COCO格式要求
    return [float(x_min), float(y_min), float(width), float(height)]


def process_image(args):
    """处理单个图像并生成COCO格式的图像信息和标注"""
    img_file, image_id, images_path, labels_path = args
    img_path = os.path.join(images_path, img_file)

    try:
        # 使用imagesize快速获取图像尺寸，而不是完全加载图像
        try:
            img_width, img_height = imagesize.get(img_path)
        except:
            # 如果imagesize失败，回退到PIL
            with Image.open(img_path) as img:
                img_width, img_height = img.size

        # 创建图像信息
        image_info = {
            "id": image_id,
            "file_name": img_file,
            "width": img_width,
            "height": img_height,
        }

        # 为递归目录结构处理标签路径
        # 处理子目录中的图像，对应的标签文件路径也需要匹配
        img_dir_rel_path = os.path.dirname(img_file)
        img_filename = os.path.basename(img_file)
        label_base_name = os.path.splitext(img_filename)[0]

        if img_dir_rel_path:
            # 图像在子目录中，对应的标签也应该在子目录中
            label_dir_path = os.path.join(labels_path, img_dir_rel_path)
            label_file = os.path.join(label_dir_path, label_base_name + ".txt")
        else:
            # 图像在根目录中
            label_file = os.path.join(labels_path, label_base_name + ".txt")

        annotations = []

        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        # YOLO坐标：中心点x,y,宽度,高度（归一化为0-1）
                        bbox = [float(coord) for coord in parts[1:5]]

                        # 转换为COCO格式的边界框
                        coco_bbox = yolo_bbox_to_coco(bbox, img_width, img_height)

                        # 计算区域面积
                        area = coco_bbox[2] * coco_bbox[3]

                        # 创建标注信息
                        annotations.append(
                            {
                                "image_id": image_id,
                                "category_id": class_id,
                                "bbox": coco_bbox,
                                "area": area,
                                "segmentation": [],
                                "iscrowd": 0,
                            }
                        )

        return image_info, annotations
    except Exception as e:
        print(f"处理图像 {img_file} 时出错: {str(e)}")
        return None, []


def yolo_to_coco(
    root_dir,
    img_dir="images",
    label_dir="labels",
    class_names_file=None,
    num_workers=4,
    recursive=False,
):
    """将YOLO格式数据集转换为COCO格式，使用多线程加速处理"""
    # 确定图像和标签路径
    images_path = os.path.join(root_dir, img_dir)
    labels_path = os.path.join(root_dir, label_dir)

    # 加载类别名称
    if class_names_file and os.path.exists(class_names_file):
        with open(class_names_file, "r") as f:
            class_names = [line.strip() for line in f.readlines()]
    else:
        raise ValueError("类别名称文件不存在")

    # 构建COCO格式数据
    coco_data = {
        "info": {
            "root_dir": root_dir,
            "img_dir": img_dir,
            "label_dir": label_dir,
            "description": "Converted from YOLO format",
            "version": "1.0",
            "year": datetime.datetime.now().year,
            "contributor": "YOLO to COCO converter",
            "date_created": datetime.datetime.now().strftime("%Y-%m-%d"),
        },
        "licenses": [{"id": 1, "name": "Unknown", "url": "Unknown"}],
        "images": [],
        "annotations": [],
        "categories": [],
    }

    # 添加类别信息
    for idx, class_name in enumerate(class_names):
        coco_data["categories"].append(
            {"id": idx, "name": class_name, "supercategory": "None"}
        )

    # 递归获取图像文件列表
    image_files = get_image_files(images_path, recursive)
    print(f"找到 {len(image_files)} 个图像文件")

    # 创建任务列表
    tasks = [
        (img_file, i, images_path, labels_path)
        for i, img_file in enumerate(image_files)
    ]

    # 使用线程池并发处理图像
    images = []
    annotations = []
    annotation_id = 0

    with tqdm(total=len(image_files), desc="处理图像") as pbar:
        # 创建线程锁，用于保护进度条更新
        lock = threading.Lock()

        # 定义回调函数
        def update_progress(*args):
            with lock:
                pbar.update(1)

        # 使用线程池并发处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []

            # 提交任务到线程池
            for task in tasks:
                future = executor.submit(process_image, task)
                future.add_done_callback(update_progress)
                futures.append(future)

            # 获取结果
            for future in concurrent.futures.as_completed(futures):
                image_info, image_annotations = future.result()
                if image_info:  # 只有处理成功的图像才添加
                    images.append(image_info)

                    for anno in image_annotations:
                        anno["id"] = annotation_id
                        annotations.append(anno)
                        annotation_id += 1

    # 按ID排序图像
    images.sort(key=lambda x: x["id"])
    coco_data["images"] = images
    coco_data["annotations"] = annotations

    return coco_data


def main():
    """主函数"""
    args = parse_args()

    # 转换数据
    coco_data = yolo_to_coco(
        args.root_dir,
        args.img_dir,
        args.label_dir,
        args.class_names,
        args.num_workers,
        args.recursive,
    )

    # 保存COCO格式数据
    with open(args.output, "w") as f:
        json.dump(coco_data, f, indent=2)

    print(f"转换完成! COCO格式数据已保存到: {args.output}")
    print(f"统计信息:")
    print(f"- 图像数量: {len(coco_data['images'])}")
    print(f"- 标注数量: {len(coco_data['annotations'])}")
    print(f"- 类别数量: {len(coco_data['categories'])}")


if __name__ == "__main__":
    main()
