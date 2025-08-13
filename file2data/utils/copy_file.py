"""
copy files with multi threads

args:
- file_txt: txt file, each line is a file path
- src_root_dir: source root directory
- dst_root_dir: destination root directory
- num_workers: number of workers
- output_txt: output txt file with new file paths (optional)

features:
- support multi threads, tqdm progress bar
- output new txt file with new file paths

demo:
python3 file2data/utils/copy_file.py \
    --file_txt /fsx/workspace_robin/datasets/anker/reid/train_data_akpu.txt \
    --src_root_dir /media/zhixineushare/work_jimmy/vlm_datasets/data/id/train_data_akpu \
    --dst_root_dir /opt/dlami/nvme/workspace_robin/datasets/anker/reid/train_data_akpu \
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

def copy_single_file(file_path, src_root_dir, dst_root_dir):
    """
    copy single file
    """
    # 构建源图片的完整路径
    src_path = os.path.join(src_root_dir, file_path.strip())
    
    # 检查源文件是否存在
    if not os.path.exists(src_path):
        return None
    
    # 构建目标图片的完整路径
    if src_path.startswith(src_root_dir):
        dst_path = src_path.replace(src_root_dir, dst_root_dir)
    else:
        raise ValueError(f"File path {src_path} is not in {src_root_dir}")
    
    if osp.exists(dst_path):
        return osp.relpath(dst_path, dst_root_dir)
    
    # 创建目标目录
    os.makedirs(osp.dirname(dst_path), exist_ok=True)
    
    # 复制文件
    try:
        os.system(f'cp "{src_path}" "{dst_path}"')
        # 返回相对于目标目录的路径
        return osp.relpath(dst_path, dst_root_dir)
    except Exception as e:
        print(f"Error copying {src_path}: {e}")
        return None

def copy_files(file_paths, src_root_dir, dst_root_dir, num_workers=4):
    """
    copy multiple files with multi-threading
    """
    func = partial(copy_single_file, src_root_dir=src_root_dir, dst_root_dir=dst_root_dir)
    
    # 使用多线程复制图片
    results = parallelise(func, file_paths, num_workers=num_workers, task_type="io_bound")
    
    # 过滤出成功复制的图片路径
    valid_paths = [path for path in results if path is not None]
    invalid_count = len(results) - len(valid_paths)
    
    return valid_paths, invalid_count

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='copy files with multi threads')
    parser.add_argument('--file_txt', type=str, required=True, help='txt file containing file paths')
    parser.add_argument('--src_root_dir', type=str, required=True, help='source root directory')
    parser.add_argument('--dst_root_dir', type=str, required=True, help='destination root directory')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for multi-threading')
    parser.add_argument('--output_txt', type=str, help='output txt file with new file paths (optional)')
    args = parser.parse_args()

    # 加载图片路径列表
    file_paths = load_file(args.file_txt)
    print(f"Found {len(file_paths)} file paths in {args.file_txt}")
    
    # 复制图片
    print(f"Copying files from {args.src_root_dir} to {args.dst_root_dir} with {args.num_workers} workers...")
    valid_paths, invalid_count = copy_files(file_paths, args.src_root_dir, args.dst_root_dir, args.num_workers)
    
    # 输出结果
    print(f"Successfully copied {len(valid_paths)} files")
    if invalid_count > 0:
        print(f"Failed to copy {invalid_count} files")
    
    # 保存新的图片路径文件
    if args.output_txt:
        save_file(args.output_txt, '\n'.join(valid_paths))
        print(f"Saved new file paths to {args.output_txt}")
