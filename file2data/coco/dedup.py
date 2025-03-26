"""
dedup similar images in coco dataset

input:
    - in_file: coco dataset json file
    - img_dir: coco dataset image directory
output:
    - out_file: deduped coco dataset

description:
    - use phash and slide-window to find duplicates
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
from imagededup.methods import PHash
import orjson
import random
from tqdm import trange, tqdm
from pybloom_live import BloomFilter

def fast_load_coco_dataset(file_path):
    with open(file_path, "r") as f:
        return orjson.loads(f.read())

def fast_write_coco_dataset(file_path, data):
    with open(file_path, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))

def parse_args():
    parser = argparse.ArgumentParser(
        description="Deduplicate similar images in COCO dataset"
    )
    parser.add_argument("--in-file", required=True, help="Input COCO dataset JSON file")
    parser.add_argument(
        "--img-dir", required=True, help="Input COCO dataset image directory"
    )
    parser.add_argument(
        "--out-file", required=True, help="Output deduped COCO dataset JSON file"
    )
    parser.add_argument(
        "--cache-file", required=False, help="Output phash cache file", default=None
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=1,
        help="Hamming distance threshold for PHash (lower = more strict, range 0-64)",
    )
    parser.add_argument('--sample_size', type=int, default=1000000, help='sample size for deduplication')
    args = parser.parse_args()

    return args


def load_coco_dataset(json_path: str) -> Dict:
    """Load COCO dataset from JSON file"""
    print(f"Loading COCO dataset from {json_path}...")
    with open(json_path, "r") as f:
        dataset = json.load(f)

    print(
        f"Dataset loaded with {len(dataset['images'])} images and {len(dataset['annotations'])} annotations"
    )
    return dataset


def get_image_paths(coco_data: Dict, img_dir: str) -> Dict[str, str]:
    """Create mapping from image filename to full path"""
    image_paths = {}
    for image in coco_data["images"]:
        filename = image["file_name"]
        image_paths[filename] = os.path.join(img_dir, filename)
    return image_paths

def hamming_distance(hash1, hash2):
    """
    Calculate Hamming distance between two hashes.

    Args:
        hash1: First hash
        hash2: Second hash

    Returns:
        distance: Hamming distance between the two hashes
    """
    return bin(int(hash1, 16) ^ int(hash2, 16)).count("1")

def bloom_sample(encodings, duplicates, sample_size=1000000):
    bloom_filter = BloomFilter(capacity=len(encodings), error_rate=0.0001)
    result = []
    for key, value in tqdm(encodings.items(), desc='Deduplicating'):
        if value not in bloom_filter:
            bloom_filter.add(value)
            if key in duplicates:
                result.append(key)

    print(f"bloom_sample: from {len(duplicates)} to {len(result)}")
    if len(result) > sample_size:
        result = random.sample(result, sample_size)
    return result

def dedup_with_sliding_window(encodings, window=30, threshold=8):
    """
    Deduplicate images using a sliding window approach to compute Hamming distances.

    Args:
        encodings: Dictionary of image paths to their phash values
        window: Size of sliding window for comparison
        threshold: Maximum Hamming distance to consider as duplicate

    Returns:
        duplicates: Dictionary mapping original images to their duplicates
    """
    # Convert encodings to sorted list of (path, hash) tuples
    sorted_encodings = sorted(encodings.items())
    n = len(sorted_encodings)

    duplicates = {}
    processed = set()  # Track processed image paths to avoid duplicates

    for i in trange(n):
        img_path, img_hash = sorted_encodings[i]

        # Skip if this image has already been marked as a duplicate
        if img_path in processed:
            continue

        # Initialize entry in duplicates dictionary
        if img_path not in duplicates:
            duplicates[img_path] = []

        # Compare with next 'window' images
        for j in range(i + 1, min(i + window + 1, n)):
            compare_path, compare_hash = sorted_encodings[j]

            # Skip already processed images
            if compare_path in processed:
                continue

            # Calculate Hamming distance between hashes
            distance = hamming_distance(img_hash, compare_hash)

            # If below threshold, consider as duplicate
            if distance <= threshold:
                duplicates[img_path].append(compare_path)
                processed.add(compare_path)  # Mark this as processed

    return duplicates

def deduplicate_coco(
    coco_data: Dict, duplicates: Dict[str, List[str]],
) -> Tuple[Dict, Set[int], Set[str]]:
    """Remove duplicates from COCO dataset"""
    # Create a set of duplicate image filenames to remove
    images_to_remove = set()
    for filename, dup_list in duplicates.items():
        # Keep the original, remove the duplicates
        for dup in dup_list:
            images_to_remove.add(dup)

    # Create mapping from filename to image_id
    filename_to_id = {img["file_name"]: img["id"] for img in coco_data["images"]}

    # Get image IDs to remove
    image_ids_to_remove = {
        filename_to_id[filename]
        for filename in images_to_remove
        if filename in filename_to_id
    }

    # Filter images
    deduped_images = [
        img for img in coco_data["images"] if img["id"] not in image_ids_to_remove
    ]

    # Filter annotations
    deduped_annotations = [
        ann
        for ann in coco_data["annotations"]
        if ann["image_id"] not in image_ids_to_remove
    ]

    # Create new COCO dataset
    deduped_coco = coco_data.copy()
    deduped_coco["images"] = deduped_images
    deduped_coco["annotations"] = deduped_annotations

    return deduped_coco, image_ids_to_remove, images_to_remove

def sample_coco(coco_data: Dict, sampled_img_paths: List[str]) -> Dict:
    """
    sample coco dataset with deduped image paths
    """
    img_path_set = set(sampled_img_paths)
    dedup_img_ids = {img['id'] for img in coco_data['images'] if img['file_name'] in img_path_set}
    deduped_images = [img for img in coco_data['images'] if img['id'] in dedup_img_ids]
    deduped_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in dedup_img_ids]
    deduped_data = {
        'images': deduped_images,
        'annotations': deduped_annotations,
        'categories': coco_data['categories']
    }
    return deduped_data

def main():
    args = parse_args()

    # Load COCO dataset
    coco_data = load_coco_dataset(args.in_file)

    # Get image paths
    # image_paths = get_image_paths(coco_data, args.img_dir)
    for img_info in coco_data['images']:
        img_info['file_name'] = os.path.join(args.img_dir, img_info['file_name'])
    image_paths = [img_info['file_name'] for img_info in coco_data['images']]

    # Find duplicates
    # duplicates = find_duplicates(image_paths, args.threshold)
    phasher = PHash()
    if args.cache_file:
        phash_cache_path = args.cache_file
    else:
        phash_cache_path = Path(args.out_file).with_suffix(".phash_cache.json")
    if os.path.exists(phash_cache_path):
        print(f'Loading phash cache from {phash_cache_path}')
        with open(phash_cache_path, "r") as f:
            encodings = orjson.loads(f.read())
    else:
        encodings = phasher.encode_images(files=image_paths)
        fast_write_coco_dataset(phash_cache_path, encodings)
        print(f"Saved phash cache to {phash_cache_path}")
    print(f"len(encodings): {len(encodings)}")

    # need 100h+ to find duplicates
    # duplicates = phasher.find_duplicates(encoding_map=encodings, max_distance_threshold=args.threshold)
    duplicates = dedup_with_sliding_window(encodings, window=30, threshold=args.threshold)
    print(f"Found {len(duplicates)} unique images")
    dup_log_file = Path(args.out_file).with_suffix(".duplicates.log")
    with open(dup_log_file, "w") as f:
        for filename, dup_list in duplicates.items():
            f.write(f"{filename}: {dup_list}\n")
    print(f"Saved duplicate log to {dup_log_file}")

    # Count total duplicates found
    total_duplicates = sum(len(dup_list) for dup_list in duplicates.values())
    print(f"Found {total_duplicates} duplicate images")

    if args.sample_size > 0:
        sampled_img_paths = bloom_sample(encodings, duplicates, sample_size=args.sample_size)
        print(f"Sampled {len(sampled_img_paths)} images for deduplication")
        deduped_coco = sample_coco(coco_data, sampled_img_paths)
        print(
            f"Original dataset: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations"
        )
        print(f"Sampled dataset: {len(deduped_coco['images'])} images, {len(deduped_coco['annotations'])} annotations")
    else:
        # Deduplicate COCO dataset
        deduped_coco, removed_ids, removed_files = deduplicate_coco(coco_data, duplicates)
        print(
            f"Original dataset: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations"
        )
        print(
            f"Deduped dataset: {len(deduped_coco['images'])} images, {len(deduped_coco['annotations'])} annotations"
        )

    # Save deduped COCO dataset
    output_dir = os.path.dirname(args.out_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.out_file, "w") as f:
        json.dump(deduped_coco, f, indent=2)
    print(f"Saved deduplicated dataset to {args.out_file}")


if __name__ == "__main__":
    main()
