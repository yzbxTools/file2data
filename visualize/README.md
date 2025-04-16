# COCO格式预测结果可视化工具

这个工具使用Streamlit提供了一个交互式界面，用于可视化COCO格式的预测结果。

## 功能特点

1. 通过图像ID选择特定图像
2. 按类别名称过滤，可选择单个或多个类别
3. 按预测分数过滤，范围从0到1
4. 分页查看，每页显示12个预测结果（4行3列）
5. 可视化边界框和预测标签

## 使用方法

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行可视化工具

有两种方式运行可视化工具：

1. 通过命令行参数指定JSON文件：

```bash
streamlit run view_prediction.py -- --file path/to/your/predictions.json
```

2. 直接运行，然后通过界面上传文件：

```bash
streamlit run view_prediction.py
```

## COCO格式要求

输入文件应为COCO格式的JSON文件，包含以下字段：

- `images`: 图像信息列表
- `categories`: 类别信息列表
- `annotations`: 预测结果列表

每个预测结果应包含：
- `image_id`: 对应的图像ID
- `category_id`: 类别ID
- `bbox`: 边界框坐标 [x, y, width, height]
- `score`: 预测分数（可选） 