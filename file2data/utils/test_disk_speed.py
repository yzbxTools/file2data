"""
test image loading speed in disk

usage:
python3 file2data/utils/test_disk_speed.py \
    --image_txt_file <image_txt_file> \
    --num_threads <num_threads> \
    --num_samples <num_samples>

random select num_samples from image_txt_file and test disk speed
"""

import os
import random
import time
import argparse
from PIL import Image
from tqdm import tqdm
from typing import List
from file2data.utils import parallelise

def load_image(img_path: str) -> None:
    """加载单张图片"""
    try:
        img = Image.open(img_path).convert('RGB')
        img.load()  # 确保完全加载到内存
    except Exception as e:
        print(f"加载图片 {img_path} 失败: {e}")


def test_disk_speed(img_list: List[str], num_samples: int, num_threads: int) -> None:
    """测试磁盘读取速度"""
    if num_samples > len(img_list):
        num_samples = len(img_list)
        
    # 随机采样
    sampled_imgs = random.sample(img_list, num_samples)
    
    start_time = time.time()
    if num_threads > 1:
        parallelise(load_image, sampled_imgs, num_workers=num_threads)
    else:
        for img_path in tqdm(sampled_imgs, desc="加载图片"):
            load_image(img_path)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_samples
    print(f"\n总耗时: {total_time:.2f}秒")
    print(f"平均每张图片耗时: {avg_time*1000:.2f}毫秒") 


def main():
    parser = argparse.ArgumentParser(description="测试磁盘图片加载速度")
    parser.add_argument("--image_txt_file", type=str, help="包含图片路径的txt文件")
    parser.add_argument("--num_samples", type=int, help="要测试的图片数量")
    parser.add_argument("--num_threads", type=int, help="线程数")
    args = parser.parse_args()

    # 读取图片列表
    with open(args.image_txt_file, "r") as f:
        img_list = [line.strip() for line in f.readlines()]
    
    test_disk_speed(img_list, args.num_samples, args.num_threads)


if __name__ == "__main__":
    main()


