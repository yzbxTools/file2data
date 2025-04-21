"""
clean the mmdet model weights, keep the latest and best checkpoint

example:
- best_coco_bbox_mAP_epoch_20.pth
- epoch_14.pth
- step_10000.pth
"""

import os
import glob
import argparse
import re
from loguru import logger

def get_newest_files(file_list):
    """
    get the newest file from the file list
    """
    return max(file_list, key=os.path.getctime)

def get_latest_ckpt(file_list):
    """
    get the latest checkpoint from the file list
    """
    if len(file_list) == 0:
        return None

    if "epoch" in file_list[0]:
        return max(file_list, key=lambda x: int(re.search(r'epoch_(\d+)', x).group(1)))
    elif "step" in file_list[0]:
        return max(file_list, key=lambda x: int(re.search(r'step_(\d+)', x).group(1)))
    else:
        return None

def clean_mmdet_weights(model_dir, recursive=False, print_only=False):
    """
    clean the mmdet model weights, keep the latest and best checkpoint
    """
    if not recursive:
        best_ckpts = glob.glob(os.path.join(model_dir, "best*.pth"))
        loop_ckpts = glob.glob(os.path.join(model_dir, "epoch_*.pth")) + glob.glob(os.path.join(model_dir, "step_*.pth"))
        if len(best_ckpts) > 0:
            newest_best_ckpt = get_newest_files(best_ckpts)
            tmp = get_latest_ckpt(best_ckpts)
            if tmp != newest_best_ckpt:
                logger.warning(f"newest_best_ckpt: {newest_best_ckpt}, tmp: {tmp}")
                newest_best_ckpt = tmp
        else:
            newest_best_ckpt = None
        if len(loop_ckpts) > 0:
            newest_loop_ckpt = get_newest_files(loop_ckpts)
            tmp = get_latest_ckpt(loop_ckpts)
            if tmp != newest_loop_ckpt:
                logger.warning(f"newest_loop_ckpt: {newest_loop_ckpt}, tmp: {tmp}")
                newest_loop_ckpt = tmp
        else:
            newest_loop_ckpt = None
        
        logger.info(f"newest_best_ckpt: {newest_best_ckpt}")
        logger.info(f"newest_loop_ckpt: {newest_loop_ckpt}")
        for ckpt in best_ckpts:
            if ckpt != newest_best_ckpt:
                if print_only:
                    print(f"remove {ckpt}")
                else:
                    os.remove(ckpt)
        for ckpt in loop_ckpts:
            if ckpt != newest_loop_ckpt:
                if print_only:
                    print(f"remove {ckpt}")
                else:
                    os.remove(ckpt)
    else:
        pth_files = glob.glob(os.path.join(model_dir, "*.pth"))
        if len(pth_files) > 0:
            clean_mmdet_weights(model_dir, recursive=False, print_only=print_only)
        else:
            sub_dirs = glob.glob(os.path.join(model_dir, "*"))
            for sub_dir in sub_dirs:
                clean_mmdet_weights(sub_dir, recursive=True, print_only=print_only)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="")
    parser.add_argument("--print_only", default=False, action="store_true")
    parser.add_argument("--recursive", default=False, action="store_true")
    args = parser.parse_args()
    clean_mmdet_weights(args.model_dir, args.recursive, args.print_only)