"""
clean the mmdet model weights, keep the latest and best checkpoint
"""

import os
import glob
import argparse

def get_newest_files(file_list):
    """
    get the newest file from the file list
    """
    return max(file_list, key=os.path.getctime)

def clean_mmdet_weights(model_dir, recursive=False, print_only=False):
    """
    clean the mmdet model weights, keep the latest and best checkpoint
    """
    if not recursive:
        best_ckpts = glob.glob(os.path.join(model_dir, "best*.pth"))
        loop_ckpts = glob.glob(os.path.join(model_dir, "epoch_*.pth")) + glob.glob(os.path.join(model_dir, "step_*.pth"))
        newest_best_ckpt = get_newest_files(best_ckpts)
        newest_loop_ckpt = get_newest_files(loop_ckpts)
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
    parser.add_argument("--print_only", type=bool, default=False, action="store_true")
    parser.add_argument("--recursive", type=bool, default=False, action="store_true")
    args = parser.parse_args()
    clean_mmdet_weights(args.model_dir, args.recursive, args.print_only)