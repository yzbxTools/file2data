"""
use streamlit to show prediction results

input:
- image_file: coco format image file, contain images and categories.
- prediction_file: coco format prediction file, contain only predictions

features:
1. filter by category names, select one or multiple
2. filter by prediction scores, range from 0 to 1
3. filter image without target predictions, yes or no
4. view page by page, show 4 rows with 3 columns per page.
"""

import streamlit as st
import json
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import argparse
import sys


def load_data(image_file, prediction_file):
    if image_file is not None and prediction_file is not None:
        with open(image_file, "r") as f:
            image_data = json.load(f)
        with open(prediction_file, "r") as f:
            prediction_data = json.load(f)

        data = {
            "images": image_data["images"],
            "categories": image_data["categories"],
            "annotations": prediction_data,
        }
        return data
    return None


def get_args():
    parser = argparse.ArgumentParser(description="COCO格式预测结果可视化")
    parser.add_argument(
        "--image_file", type=str, help="COCO格式JSON文件路径, 包含images和categories"
    )
    parser.add_argument(
        "--prediction_file", type=str, help="COCO格式JSON文件路径, 只包含predictions"
    )
    args = parser.parse_args()
    return args


# 设置页面配置
st.set_page_config(
    page_title="预测结果可视化",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 标题
st.title("COCO格式预测结果可视化")

# 侧边栏 - 文件上传和过滤选项
with st.sidebar:
    st.header("数据输入")

    # 文件路径输入
    image_file = st.text_input("输入COCO格式JSON文件路径, 包含images和categories")
    prediction_file = st.text_input("输入COCO格式JSON文件路径, 只包含predictions")

    # 或者通过命令行参数传入
    if len(sys.argv) > 1:
        args = get_args()
        image_file = args.image_file
        prediction_file = args.prediction_file

    # 加载数据
    if image_file and prediction_file:
        data = load_data(image_file, prediction_file)
    else:
        data = None
        st.info("请输入COCO格式的JSON文件或通过命令行参数指定文件路径")

# 主界面
if data is not None:
    # 提取类别信息
    categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
    category_names = list(categories.values())

    # 提取图像信息
    images = {img["id"]: img for img in data.get("images", [])}
    image_ids = list(images.keys())

    # 提取预测结果
    predictions = data.get("annotations", [])
    img_id2preds: dict = {}
    for pred in predictions:
        img_id = pred["image_id"]
        if img_id not in img_id2preds:
            img_id2preds[img_id] = []
        img_id2preds[img_id].append(pred)

    # 侧边栏 - 过滤选项
    with st.sidebar:
        st.header("过滤选项")

        # 类别过滤
        selected_categories = st.multiselect(
            "选择类别", options=category_names, default=category_names
        )

        # 分数范围过滤
        min_score, max_score = st.slider(
            "预测分数范围", min_value=0.0, max_value=1.0, value=(0.0, 1.0), step=0.05
        )

        # 是否过滤无目标预测
        filter_no_target = st.checkbox("过滤无目标预测", value=True)

    # 过滤预测结果
    filtered_predictions = [
        pred
        for pred in predictions
        if categories.get(pred["category_id"], "") in selected_categories
        and min_score <= pred.get("score", 1.0) <= max_score
    ]

    if filter_no_target:
        filtered_image_ids = list(
            set([pred["image_id"] for pred in filtered_predictions])
        )
    else:
        filtered_image_ids = image_ids

    # 分页显示
    items_per_page = 12  # 4行3列
    total_pages = max(
        1,
        len(filtered_image_ids) // items_per_page
        + (1 if len(filtered_image_ids) % items_per_page > 0 else 0),
    )

    current_page = st.sidebar.number_input(
        "页码", min_value=1, max_value=total_pages, value=1
    )

    # 显示当前页的预测结果
    start_idx = (current_page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(filtered_image_ids))
    current_image_ids = filtered_image_ids[start_idx:end_idx]

    # 创建4行3列的网格布局
    st.subheader(f"预测结果 (第 {current_page}/{total_pages} 页)")

    # 使用列布局创建网格
    for row in range(4):
        cols = st.columns(3)
        for col in range(3):
            idx = row * 3 + col
            if idx < len(current_image_ids):
                selected_image_id = current_image_ids[idx]
                image_info = images[selected_image_id]
                image_path = image_info.get("file_name", "")

                with cols[col]:
                    # 尝试加载图像
                    try:
                        if os.path.exists(image_path):
                            img = Image.open(image_path)
                            draw = ImageDraw.Draw(img)
                        else:
                            # 如果找不到图像，创建一个空白图像
                            img = Image.new(
                                "RGB",
                                (
                                    image_info.get("width", 800),
                                    image_info.get("height", 600),
                                ),
                                color="white",
                            )
                            draw = ImageDraw.Draw(img)
                            draw.text((10, 10), f"图像未找到: {image_path}", fill="red")

                        # 显示预测结果
                        if selected_image_id in img_id2preds:
                            current_predictions = [
                                pred
                                for pred in img_id2preds[selected_image_id]
                                if categories.get(pred["category_id"], "")
                                in selected_categories
                                and min_score <= pred.get("score", 1.0) <= max_score
                            ]

                            # draw bbox on image

                            for pred in current_predictions:
                                bbox = pred.get("bbox", [0, 0, 0, 0])  # x,y,w,h
                                bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]  # x1,y1,x2,y2
                                draw.rectangle(bbox, outline="red", width=2)
                                category_name = categories.get(
                                    pred["category_id"], "Unknown"
                                )
                                score = pred.get("score", 1.0)
                                draw.text(
                                    (bbox[0], bbox[1]),
                                    f"{category_name}: {score:.2f}",
                                    fill="red",
                                )

                            st.image(
                                img,
                                caption=f"图像ID: {selected_image_id}",
                                use_container_width=True,
                            )

                    except Exception as e:
                        st.error(f"处理图像时出错: {str(e)}")

# 如果没有数据，显示使用说明
else:
    st.markdown(
        """
    ## 使用说明

    1. 上传COCO格式的JSON文件，或通过命令行参数指定文件路径
    2. 使用侧边栏的过滤选项筛选预测结果:
       - 选择一个或多个类别
       - 设置预测分数范围
       - 是否过滤无目标预测
    3. 使用分页控件浏览预测结果
    4. 查看图像上的预测结果表格
    """
    )

# 命令行参数解析
if __name__ == "__main__":
    args = get_args()

    if args.prediction_file and args.image_file:
        if os.path.exists(args.prediction_file) and os.path.exists(args.image_file):
            # 命令行参数已处理，Streamlit会自动运行应用程序
            pass
        else:
            print(f"文件不存在")
            sys.exit(1)
