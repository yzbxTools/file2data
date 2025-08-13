"""
copy image files with multi threads

args:
- img_txt: txt file, each line is a image file path
- src_img_dir: source image directory
- dst_img_dir: destination image directory
- num_workers: number of workers
- output_txt: output txt file with new image paths (optional)

features:
- support multi threads, tqdm progress bar
- output new txt file with new image file paths

demo:
python3 file2data/utils/copy_img.py \
    --img_txt /fsx/workspace_robin/datasets/anker/reid/train_data_akpu.txt \
    --src_img_dir /media/zhixineushare/work_jimmy/vlm_datasets/data/id/train_data_akpu \
    --dst_img_dir /opt/dlami/nvme/workspace_robin/datasets/anker/reid/train_data_akpu \
    --num_workers 16 \
    --output_txt /opt/dlami/nvme/workspace_robin/datasets/anker/reid/train_data_akpu.txt
"""

from file2data import load_file, save_file
from file2data.utils import parallelise
from functools import partial
import argparse 
import os 
import os.path as osp
from tqdm import tqdm

def copy_single_img(img_path, src_img_dir, dst_img_dir):
    """
    copy single image file
    """
    # 构建源图片的完整路径
    src_img_path = os.path.join(src_img_dir, img_path.strip())
    
    # 检查源文件是否存在
    if not os.path.exists(src_img_path):
        return None
    
    # 构建目标图片的完整路径
    if src_img_path.startswith(src_img_dir):
        dst_img_path = src_img_path.replace(src_img_dir, dst_img_dir)
    else:
        raise ValueError(f"Image path {src_img_path} is not in {src_img_dir}")
    
    if osp.exists(dst_img_path):
        return osp.relpath(dst_img_path, dst_img_dir)
    
    # 创建目标目录
    os.makedirs(osp.dirname(dst_img_path), exist_ok=True)
    
    # 复制文件
    try:
        os.system(f'cp "{src_img_path}" "{dst_img_path}"')
        # 返回相对于目标目录的路径
        return osp.relpath(dst_img_path, dst_img_dir)
    except Exception as e:
        print(f"Error copying {src_img_path}: {e}")
        return None

def copy_images(img_paths, src_img_dir, dst_img_dir, num_workers=4):
    """
    copy multiple image files with multi-threading
    """
    func = partial(copy_single_img, src_img_dir=src_img_dir, dst_img_dir=dst_img_dir)
    
    # 使用多线程复制图片
    results = parallelise(func, img_paths, num_workers=num_workers, task_type="io_bound")
    
    # 过滤出成功复制的图片路径
    valid_paths = [path for path in results if path is not None]
    invalid_count = len(results) - len(valid_paths)
    
    return valid_paths, invalid_count

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='copy image files with multi threads')
    parser.add_argument('--img_txt', type=str, required=True, help='txt file containing image paths')
    parser.add_argument('--src_img_dir', type=str, required=True, help='source image directory')
    parser.add_argument('--dst_img_dir', type=str, required=True, help='destination image directory')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for multi-threading')
    parser.add_argument('--output_txt', type=str, help='output txt file with new image paths (optional)')
    args = parser.parse_args()

    # 加载图片路径列表
    img_paths = load_file(args.img_txt)
    print(f"Found {len(img_paths)} image paths in {args.img_txt}")
    
    # 复制图片
    print(f"Copying images from {args.src_img_dir} to {args.dst_img_dir} with {args.num_workers} workers...")
    valid_paths, invalid_count = copy_images(img_paths, args.src_img_dir, args.dst_img_dir, args.num_workers)
    
    # 输出结果
    print(f"Successfully copied {len(valid_paths)} images")
    if invalid_count > 0:
        print(f"Failed to copy {invalid_count} images")
    
    # 保存新的图片路径文件
    if args.output_txt:
        save_file(args.output_txt, '\n'.join(valid_paths))
        print(f"Saved new image paths to {args.output_txt}")
