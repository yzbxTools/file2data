# FP/FN结果可视化工具

这是一个用于可视化目标检测结果中假阳性(FP)和假阴性(FN)的工具。该工具基于Streamlit构建，可以直观地展示目标检测模型的性能。

## 功能特点

1. 校准预测分数阈值（0-1范围）
2. 根据预测分数分类FP/FN（0-1范围）
3. 按类别名称过滤，可选择单个或多个类别
4. 过滤没有目标预测的图像
5. 按FP/FN过滤
6. 分页查看，每页显示4行2列（左侧为GT，右侧为预测）
7. 支持通过命令行参数传递参数

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/file2data.git
cd file2data
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

### 通过命令行运行

```bash
streamlit run visualize/view_fp_fn.py -- --image_file path/to/image_file.json --prediction_file path/to/prediction_file.json
```

### 通过Web界面运行

```bash
streamlit run visualize/view_fp_fn.py
```

然后在浏览器中打开显示的URL，通过界面上传COCO格式的JSON文件。

## 输入文件格式

- **图像文件**：COCO格式的JSON文件，包含images、categories和gt annotations
- **预测文件**：COCO格式的JSON文件，只包含predictions

## 使用说明

1. 在侧边栏中设置预测分数阈值和IoU阈值
2. 选择要显示的类别
3. 选择是否显示没有目标预测的图像
4. 选择是否显示假阳性(FP)和假阴性(FN)
5. 使用页码滑块浏览结果

## 注意事项

- 确保图像文件路径正确，且图像文件可访问
- 预测文件中的类别ID应与图像文件中的类别ID一致

## coco
- 