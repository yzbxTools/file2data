"""
copy image files for coco dataset 

usage:
    python3 file2data/coco/coco_img.py \
        --src_ann src.json \
        --src_img_dir src_img_dir \
        --dst_ann des.json \
        --dst_img_dir dst_img_dir 
"""

from file2data import load_json, save_json
from file2data.utils import parallelise
from functools import partial
import argparse 
import os 
import os.path as osp

def copy_img(img_info, src_img_dir, dst_img_dir):
    """
    copy image files
    """
    src_img_path = os.path.join(src_img_dir, img_info['file_name'])
    if src_img_path.startswith(src_img_dir):
        dst_img_path = src_img_path.replace(src_img_dir, dst_img_dir)
        if not osp.exists(dst_img_path):
            os.makedirs(osp.dirname(dst_img_path), exist_ok=True)
            os.system(f'cp "{src_img_path}" "{dst_img_path}"')
        img_info['file_name'] = osp.relpath(dst_img_path, dst_img_dir)
        return img_info
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='copy image files for coco dataset')
    parser.add_argument('--src_ann', type=str, required=True, help='source annotation file')
    parser.add_argument('--src_img_dir', type=str, required=True, help='source image directory')
    parser.add_argument('--dst_ann', type=str, required=True, help='destination annotation file')
    parser.add_argument('--dst_img_dir', type=str, required=True, help='destination image directory')
    args = parser.parse_args()

    # load json
    src_ann = load_json(args.src_ann)

    des_ann = {'images': [], 'annotations': [], 'categories': src_ann['categories']}
    # copy images
    func = partial(copy_img, src_img_dir=args.src_img_dir, dst_img_dir=args.dst_img_dir)
    imgs = parallelise(func, src_ann['images'])

    valid_imgs = [img for img in imgs if img is not None]
    invalid_img_num = len(imgs) - len(valid_imgs)
    valid_img_ids = set([img['id'] for img in valid_imgs])
    valid_anns = [ann for ann in src_ann['annotations'] if ann['image_id'] in valid_img_ids]

    invalid_img_names = [img['file_name'] for img in imgs if img is None]
    if invalid_img_num > 0:
        print(f'Warning: {invalid_img_num} images are not copied. Please check the source image directory.')
        print(f'Invalid image names[0:5]: {invalid_img_names[0:5]}')
    
    des_ann['images'] = valid_imgs
    des_ann['annotations'] = valid_anns
    # save json
    save_json(args.dst_ann, des_ann)
    print(f'Copy {len(valid_imgs)} images and {len(valid_anns)} annotations to {args.dst_ann}')
    print(f'Copy {len(valid_imgs)} images to {args.dst_img_dir}')