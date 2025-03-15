import json
import argparse
from typing import Dict

def merge_annotations(coco_file1: str, coco_file2: str, output_file: str) -> None:
    """
    Merge annotations from two COCO JSON files.

    Args:
        coco_file1 (str): Path to the first COCO JSON file.
        coco_file2 (str): Path to the second COCO JSON file.
        output_file (str): Path to the output merged COCO JSON file.
    """
    with open(coco_file1, 'r') as f1, open(coco_file2, 'r') as f2:
        coco1 = json.load(f1)
        coco2 = json.load(f2)

    merged_coco = coco1
    merged_coco['annotations'].extend(coco2['annotations'])

    with open(output_file, 'w') as f:
        json.dump(merged_coco, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge annotations from two COCO JSON files.")
    parser.add_argument("coco_file1", type=str, help="Path to the first COCO JSON file")
    parser.add_argument("coco_file2", type=str, help="Path to the second COCO JSON file")
    parser.add_argument("output_file", type=str, help="Path to the output merged COCO JSON file")
    args = parser.parse_args()

    merge_annotations(args.coco_file1, args.coco_file2, args.output_file)