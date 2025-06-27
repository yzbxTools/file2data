"""
restore fsx file system (lustre) with s3 (init s3 data in fsx)
release fsx file system (lustre) to s3 (release fsx data)
archive fsx file system (lustre) to s3 (export fsx data back to s3)

usage:
python3 file2data/aws/play_fsx.py \
    --fsx_dir <fsx_img_root> \
    --num_workers <num_workers> \
    --app <app> \
    --sudo

demo:
python3 file2data/aws/play_fsx.py \
    --fsx_dir /fsx/fsx_ap/workspace_robin/experiments/yolo_world/zx5 \
    --sudo \
    --app archive
"""

import argparse
import os.path as osp
import os
from file2data.utils import parallelise
import subprocess
from functools import partial


def lfs_play(file_path: str, sudo: bool, app: str) -> dict:
    """
    init: init s3 file content to fsx
    release: lfs release file content
    sync: lfs archive file_path to s3"""

    if app in ['init', 'restore', 'import']:
        cmd = f"lfs hsm_restore '{file_path}'"
    elif app in ['release']:
        cmd = f"lfs hsm_release '{file_path}'"
    elif app in ['archive', 'export']:
        cmd = f"lfs hsm_archive '{file_path}'"
    else:
        raise ValueError(f"invalid app: {app}")

    if sudo:
        cmd = f"sudo {cmd}"
    try:
        subprocess.run(cmd, shell=True, check=True)
    except Exception as e:
        print(f"error: {e}")
        return dict(success=False, file_path=file_path, error=str(e))
    return dict(success=True, file_path=file_path)

def play_fsx(fsx_dir: str, app: str, sudo: bool, num_workers: int) -> None:
    """将fsx文件系统（lustre）中文件同步到s3"""
    fsx_files = []
    for root, dirs, files in os.walk(fsx_dir):
        for file in files:
            fsx_files.append(osp.join(root, file))

    if sudo:
        fn = partial(lfs_play, sudo=True, app=app)
    else:
        fn = partial(lfs_play, sudo=False, app=app)

    results = parallelise(fn, fsx_files, num_workers=num_workers)
    success_count = 0
    total_count = len(results)
    failed_files = []
    for result in results:
        if result['success']:
            success_count += 1
        else:
            failed_files.append(result['file_path'])
            if len(failed_files) < 3:
                print(f"error: {result['error']}, file_path: {result['file_path']}")

    if total_count > 0:
        success_rate = round(success_count / total_count, 4)
        print(f"success_count: {success_count}, total_count: {total_count}, success_rate: {success_rate}")
    else:
        print("No files found to process.")

    if len(failed_files) > 0:
        with open('failed_files.txt', 'w') as f:
            for file in failed_files:
                f.write(file + '\n')
        print(f"failed_files saved to failed_files.txt, total_failed_count: {len(failed_files)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsx_dir", type=str, default="/", help="fsx directory to play")
    parser.add_argument("--app", type=str, default="init", choices=['init', 'release', 'restore', 'archive', 'export', 'import'], help="app name, init=restore=import, archive=export")
    parser.add_argument("--sudo", action='store_true', help="use sudo to run lfs commands")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count()//2, help="number of workers")
    args = parser.parse_args()

    play_fsx(args.fsx_dir, args.app, args.sudo, args.num_workers)