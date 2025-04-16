"""
use streamlit to show fp/fn results

input:
- image_file: coco format image file, contain images, categories and gt annotations.
- prediction_file: coco format prediction file, contain only predictions

features:
1. calibrate the threshold of prediction scores, range from 0 to 1
2. classify fp/fn by prediction scores, range from 0 to 1
3. filter by category names, select one or multiple
4. filter image without target predictions, yes or no
5. filter by fp/fn, yes or no
6. view page by page, show 4 rows with 2 columns per page. left column is gt, right column is pred.
7. support argparse to pass parameters
"""

import streamlit as st
import argparse
import sys
import json
import os
import numpy as np
import cv2
from PIL import Image
import io
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import defaultdict


def get_args():
    parser = argparse.ArgumentParser(description="FP/FN结果可视化")
    parser.add_argument(
        "--image_file",
        type=str,
        help="COCO格式JSON文件路径, 包含images, categories和gt annotations",
    )
    parser.add_argument(
        "--prediction_file", type=str, help="COCO格式JSON文件路径, 只包含predictions"
    )
    args = parser.parse_args()
    return args


def load_data(image_file, prediction_file):
    if image_file is not None and prediction_file is not None:
        with open(image_file, "r") as f:
            image_data = json.load(f)
        with open(prediction_file, "r") as f:
            prediction_data = json.load(f)

        data = {
            "images": image_data["images"],
            "categories": image_data["categories"],
            "predictions": prediction_data,
            "gt_annotations": image_data["annotations"],
        }
        return data
    else:
        st.error("请输入COCO格式的JSON文件或通过命令行参数指定文件路径")
        sys.exit(1)


def get_category_names(categories):
    """获取所有类别名称"""
    return [cat["name"] for cat in categories]


def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # 计算交集区域的坐标
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    # 计算交集面积
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # 计算两个框的面积
    box1_area = w1 * h1
    box2_area = w2 * h2

    # 计算IoU
    iou = intersection_area / (box1_area + box2_area - intersection_area)
    return iou


def classify_fp_fn(gt_annotations, predictions, iou_threshold=0.5):
    """将预测结果分类为TP、FP和FN
    Args:
        gt_annotations: list of dict, gt annotations
        predictions: list of dict, predictions
        iou_threshold: float, iou threshold
    Returns:
        results: dict, results
        results["bbox"]: dict, bbox tp, fp, fn results
        results["image"]: dict, image tp, fp, fn results

    note: please ensure the score in predictions > the score_threshold
    the tp, fp, fn will change when the score_threshold changes
    """
    tp = []
    fp = []
    fn = []

    # 获取所有图像ID
    image_ids = set()
    for ann in gt_annotations:
        image_ids.add(ann["image_id"])
    for pred in predictions:
        image_ids.add(pred["image_id"])

    img_id2gt_anns = defaultdict(list)
    img_id2preds = defaultdict(list)
    for ann in gt_annotations:
        img_id2gt_anns[ann["image_id"]].append(ann)
    for pred in predictions:
        img_id2preds[pred["image_id"]].append(pred)

    for image_id in image_ids:
        # 获取当前图像的GT和预测
        gt_anns = img_id2gt_anns[image_id]
        preds = img_id2preds[image_id]

        # 标记已匹配的GT和预测
        matched_gt = [False] * len(gt_anns)
        matched_pred = [False] * len(preds)

        # 对每个预测，找到最佳匹配的GT
        for i, pred in enumerate(preds):
            best_iou = 0
            best_gt_idx = -1

            for j, gt in enumerate(gt_anns):
                if matched_gt[j]:
                    continue

                if pred["category_id"] != gt["category_id"]:
                    continue

                iou = calculate_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= iou_threshold:
                tp.append(pred)
                matched_gt[best_gt_idx] = True
                matched_pred[i] = True
            else:
                fp.append(pred)
                matched_pred[i] = True

        # 未匹配的GT为FN
        for j, gt in enumerate(gt_anns):
            if not matched_gt[j]:
                fn.append(gt)

    # tp_images: the images with tp bbox
    # fp_images: the images with fp bbox
    # fn_images: the images with fn bbox
    tp_images = list(set([bbox["image_id"] for bbox in tp]))
    fp_images = list(set([bbox["image_id"] for bbox in fp]))
    fn_images = list(set([bbox["image_id"] for bbox in fn]))
    results = {
        "bbox": {"tp": tp, "fp": fp, "fn": fn},
        "image": {"tp": tp_images, "fp": fp_images, "fn": fn_images},
    }
    return results


def draw_bbox(image, bbox, category_name, color, score=None):
    """在图像上绘制边界框"""
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)

    # 绘制边界框
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    # 绘制标签
    label = category_name
    if score is not None:
        label += f": {score:.2f}"

    # 计算文本大小
    font_size = 2
    (text_width, text_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_size, 1
    )

    # 绘制标签背景
    cv2.rectangle(image, (x, y - text_height - 5), (x + text_width, y), color, -1)

    # 绘制标签文本
    cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 1)

    return image


def visualize_image(
    image_path, gt_annotations, predictions, categories, score_threshold=0.5
):
    """可视化图像及其标注和预测结果"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        st.error(f"无法读取图像: {image_path}")
        return None, None

    # 转换为RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 创建GT和预测的可视化
    gt_image = image.copy()
    pred_image = image.copy()

    # 获取类别ID到名称的映射
    category_id_to_name = {cat["id"]: cat["name"] for cat in categories}

    # 绘制GT标注
    for ann in gt_annotations:
        category_name = category_id_to_name.get(ann["category_id"], "unknown")
        gt_image = draw_bbox(gt_image, ann["bbox"], category_name, (0, 255, 0))

    # 绘制预测结果
    for pred in predictions:
        if pred["score"] < score_threshold:
            continue

        category_name = category_id_to_name.get(pred["category_id"], "unknown")
        pred_image = draw_bbox(
            pred_image, pred["bbox"], category_name, (255, 0, 0), pred["score"]
        )

    return gt_image, pred_image


def main():
    # 设置页面配置
    st.set_page_config(
        page_title="FP/FN结果可视化",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("FP/FN结果可视化")

    # 获取命令行参数
    args = get_args()

    # 文件路径输入
    if args.image_file is None or args.prediction_file is None:
        st.sidebar.header("输入文件路径")
        image_file_path = st.sidebar.text_input("输入图像文件路径", type="default")
        prediction_file_path = st.sidebar.text_input("输入预测文件路径", type="default")
    else:
        image_file_path = args.image_file
        prediction_file_path = args.prediction_file

    # 加载数据
    data = load_data(image_file_path, prediction_file_path)

    # 侧边栏设置
    st.sidebar.header("设置")

    # 1. 阈值校准
    score_threshold = st.sidebar.slider("预测分数阈值", 0.0, 1.0, 0.5, 0.05)

    # 2. IoU阈值
    iou_threshold = st.sidebar.slider("IoU阈值", 0.0, 1.0, 0.5, 0.05)

    # 3. 类别过滤
    category_names = get_category_names(data["categories"])
    selected_categories = st.sidebar.multiselect(
        "选择类别", category_names, default=category_names
    )

    # 4. 图像过滤选项
    all_image_ids = set()
    for img in data["images"]:
        all_image_ids.add(img["id"])
    show_images_without_predictions = st.sidebar.checkbox(
        "显示没有目标预测的图像", value=False
    )

    # 5. FP/FN过滤
    show_fp = st.sidebar.checkbox("显示假阳性(FP)", value=True)
    show_fn = st.sidebar.checkbox("显示假阴性(FN)", value=True)
    show_tp = st.sidebar.checkbox("显示真阳性(TP)", value=True)

    # 6. create index for predictions and annotations
    img_id2info = defaultdict(dict)
    for img in data["images"]:
        img_id2info[img["id"]] = img
    img_id2preds = defaultdict(list)
    img_id2anns = defaultdict(list)
    for pred in data["predictions"]:
        img_id2preds[pred["image_id"]].append(pred)
    for ann in data["gt_annotations"]:
        img_id2anns[ann["image_id"]].append(ann)

    # 按类别与score 过滤预测结果
    selected_cat_ids = [
        cat["id"] for cat in data["categories"] if cat["name"] in selected_categories
    ]
    filtered_predictions = [
        pred
        for pred in data["predictions"]
        if pred["score"] >= score_threshold
        and pred["category_id"] in selected_cat_ids
    ]
    filtered_annotations = [
        ann for ann in data["gt_annotations"] if ann["category_id"] in selected_cat_ids
    ]

    if not show_images_without_predictions:
        show_image_ids = all_image_ids
    else:
        show_image_ids = set([p["image_id"] for p in filtered_predictions]) | set(
            [ann["image_id"] for ann in filtered_annotations]
        )

    # 分类FP/FN
    fp_fn_results = classify_fp_fn(
        filtered_annotations, filtered_predictions, iou_threshold
    )
    fp_fn_image_ids = set()
    if show_tp:
        fp_fn_image_ids.update(fp_fn_results["image"]["tp"])
    if show_fp:
        fp_fn_image_ids.update(fp_fn_results["image"]["fp"])
    if show_fn:
        fp_fn_image_ids.update(fp_fn_results["image"]["fn"])

    # 过滤图像
    filtered_image_ids = list(fp_fn_image_ids & show_image_ids)

    # show the number of tp, fp, fn for image and bbox in table
    st.subheader(f"TP/FP/FN统计 with score threshold {score_threshold}")
    st.write(f"total images: {len(filtered_image_ids)}")
    st.write(f"total gt bboxes: {len(filtered_annotations)}")
    st.write(f"total pred bboxes: {len(filtered_predictions)}")
    for key, value in fp_fn_results.items():
        tp_count = len(value["tp"])
        fp_count = len(value["fp"])
        fn_count = len(value["fn"])
        st.write(f"{key}: tp: {tp_count}, fp: {fp_count}, fn: {fn_count}")
        precision = tp_count / (tp_count + fp_count + 1e-6)
        recall = tp_count / (tp_count + fn_count + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        st.write(f"precision: {precision:.2f}, recall: {recall:.2f}, f1: {f1:.2f}")

    # 分页显示
    images_per_page = 4  # 4行2列
    total_pages = (len(filtered_image_ids) + images_per_page - 1) // images_per_page

    if total_pages > 0:
        current_page = st.sidebar.slider("页码", 1, total_pages, 1)

        start_idx = (current_page - 1) * images_per_page
        end_idx = min(start_idx + images_per_page, len(filtered_image_ids))

        current_image_ids = filtered_image_ids[start_idx:end_idx]

        # 显示图像
        for image_id in current_image_ids:
            col1, col2 = st.columns(2)
            image_info = img_id2info[image_id]
            gt_anns = [
                ann
                for ann in img_id2anns[image_id]
                if ann["category_id"] in selected_cat_ids
            ]
            preds = [
                pred
                for pred in img_id2preds[image_id]
                if pred["score"] >= score_threshold
                and pred["category_id"] in selected_cat_ids
            ]

            gt_image, pred_image = visualize_image(
                image_info["file_name"],
                gt_anns,
                preds,
                data["categories"],
                score_threshold,
            )
            # 第一列：GT
            with col1:
                if gt_image is not None:
                    st.image(gt_image, use_container_width=True)

            # 第二列：预测
            with col2:
                if pred_image is not None:
                    st.image(pred_image, use_container_width=True)
    else:
        st.warning("没有符合条件的图像")


if __name__ == "__main__":
    main()
