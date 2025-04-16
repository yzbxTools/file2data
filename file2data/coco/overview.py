"""
coco数据集概览
数量概览：
  - 图片： 总数， 正样本，负样本， 每个类别的数量及比例
  - bbox: 总数， 每个类别的数量及比例， 面积统计（最小，最大，平均，中位数）
  - 类别：总数
"""

import argparse
import os
import os.path as osp
import pandas as pd
from file2data import load_json


def parse_args():
    parser = argparse.ArgumentParser(description="coco数据集概览")
    parser.add_argument("coco_file", type=str, help="coco数据集文件")
    return parser.parse_args()


def analyze_coco(coco_data):
    """分析COCO数据集并返回统计信息"""
    # 检查数据完整性
    required_keys = ["images", "annotations", "categories"]
    for key in required_keys:
        if key not in coco_data:
            raise ValueError(f"COCO数据集缺少 '{key}' 字段")

    results = {}

    # 类别统计
    categories = coco_data["categories"]
    category_map = {cat["id"]: cat["name"] for cat in categories}
    results["类别总数"] = len(categories)
    results["类别列表"] = [f"{cat['id']}: {cat['name']}" for cat in categories]

    # 图片统计
    images = coco_data["images"]
    results["图片总数"] = len(images)

    # 图片根目录统计
    img_root_map = {}
    for img in images:
        if osp.isabs(img["file_name"]):
            # first two level directory
            seps = osp.dirname(img['file_name']).split(osp.sep)
            img_root = osp.sep.join(seps[:3])
        else:
            img_root = '.'
        if img_root not in img_root_map:
            img_root_map[img_root] = 0
        img_root_map[img_root] += 1
    results["图片根目录统计"] = img_root_map

    # 标注统计
    annotations = coco_data["annotations"]
    results["标注总数"] = len(annotations)

    # 图片ID到标注的映射
    img_to_anns = {}
    for ann in annotations:
        img_id = ann["image_id"]
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    # 正样本和负样本统计
    img_ids = set(img["id"] for img in images)
    annotated_img_ids = set(img_to_anns.keys())

    results["正样本数量"] = len(annotated_img_ids)
    results["负样本数量"] = len(img_ids - annotated_img_ids)
    results["正样本比例"] = f"{100 * len(annotated_img_ids) / len(img_ids):.2f}%"

    # 每个类别的标注数量统计
    category_counts = {}
    for ann in annotations:
        cat_id = ann["category_id"]
        if cat_id in category_map:
            cat_name = category_map[cat_id]
            if cat_name not in category_counts:
                category_counts[cat_name] = 0
            category_counts[cat_name] += 1

    results["类别标注分布"] = {}
    for cat_name, count in category_counts.items():
        results["类别标注分布"][cat_name] = {
            "数量": count,
            "比例": f"{100 * count / len(annotations):.2f}%",
        }

    # BBox面积统计
    if annotations and "bbox" in annotations[0]:
        areas = []
        for ann in annotations:
            # 如果直接包含area字段，则使用它
            if "area" in ann:
                area = ann["area"]
            # 否则从bbox计算面积
            elif "bbox" in ann:
                # COCO格式的bbox是[x,y,width,height]
                bbox = ann["bbox"]
                area = bbox[2] * bbox[3]
            else:
                continue
            areas.append(area)

        if areas:
            results["bbox面积统计"] = {
                "最小": min(areas),
                "最大": max(areas),
                "平均": sum(areas) / len(areas),
                "中位数": sorted(areas)[len(areas) // 2],
            }

    # 每个类别的图片数量统计
    img_categories = {}
    for img_id, anns in img_to_anns.items():
        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id in category_map:
                cat_name = category_map[cat_id]
                if cat_name not in img_categories:
                    img_categories[cat_name] = set()
                img_categories[cat_name].add(img_id)

    results["类别图片分布"] = {}
    for cat_name, img_ids in img_categories.items():
        results["类别图片分布"][cat_name] = {
            "数量": len(img_ids),
            "比例": f"{100 * len(img_ids) / len(images):.2f}%",
        }

    return results


def print_results(results):
    """美观地打印结果"""
    print("=" * 50)
    print("COCO数据集概览")
    print("=" * 50)

    print(f"\n📊 基本统计")
    print(f"类别总数: {results['类别总数']}")
    print(f"图片总数: {results['图片总数']}")
    print(f"标注总数: {results['标注总数']}")
    print(f"图片根目录统计: {results['图片根目录统计']}")

    print(f"\n📸 图片分析")
    print(f"正样本数量: {results['正样本数量']}")
    print(f"负样本数量: {results['负样本数量']}")
    print(f"正样本比例: {results['正样本比例']}")

    if "bbox面积统计" in results:
        print(f"\n📏 边界框面积统计")
        for key, value in results["bbox面积统计"].items():
            print(f"{key}: {value:.2f}")

    print(f"\n🏷️ 类别列表")
    for cat in results["类别列表"]:
        print(f"  - {cat}")

    print(f"\n📊 类别标注分布")
    for cat_name, stats in results["类别标注分布"].items():
        print(f"  - {cat_name}: {stats['数量']} ({stats['比例']})")

    print(f"\n🖼️ 各类别图片分布")
    for cat_name, stats in results["类别图片分布"].items():
        print(f"  - {cat_name}: {stats['数量']} ({stats['比例']})")

    print("\n" + "=" * 50)


def export_to_excel(results, output_file):
    """将结果导出为Excel文件"""
    with pd.ExcelWriter(output_file) as writer:
        # 基本信息表
        basic_info = pd.DataFrame(
            {
                "指标": [
                    "类别总数",
                    "图片总数",
                    "标注总数",
                    "正样本数量",
                    "负样本数量",
                    "正样本比例",
                    "图片根目录统计",
                ],
                "值": [
                    results["类别总数"],
                    results["图片总数"],
                    results["标注总数"],
                    results["正样本数量"],
                    results["负样本数量"],
                    results["正样本比例"],
                    results["图片根目录统计"],
                ],
            }
        )
        basic_info.to_excel(writer, sheet_name="基本信息", index=False)

        # 类别信息表
        categories_df = pd.DataFrame({"类别": results["类别列表"]})
        categories_df.to_excel(writer, sheet_name="类别列表", index=False)

        # 类别标注分布
        cat_ann_data = []
        for cat_name, stats in results["类别标注分布"].items():
            cat_ann_data.append(
                {"类别": cat_name, "标注数量": stats["数量"], "占比": stats["比例"]}
            )
        if cat_ann_data:
            cat_ann_df = pd.DataFrame(cat_ann_data)
            cat_ann_df.to_excel(writer, sheet_name="类别标注分布", index=False)

        # 类别图片分布
        cat_img_data = []
        for cat_name, stats in results["类别图片分布"].items():
            cat_img_data.append(
                {"类别": cat_name, "图片数量": stats["数量"], "占比": stats["比例"]}
            )
        if cat_img_data:
            cat_img_df = pd.DataFrame(cat_img_data)
            cat_img_df.to_excel(writer, sheet_name="类别图片分布", index=False)

        # bbox面积统计
        if "bbox面积统计" in results:
            area_data = []
            for metric, value in results["bbox面积统计"].items():
                area_data.append({"指标": metric, "值": value})
            area_df = pd.DataFrame(area_data)
            area_df.to_excel(writer, sheet_name="边界框面积统计", index=False)


if __name__ == "__main__":
    args = parse_args()
    coco_data = load_json(args.coco_file)

    # 分析数据
    results = analyze_coco(coco_data)

    # 打印结果
    print_results(results)
