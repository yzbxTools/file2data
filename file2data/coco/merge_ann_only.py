import argparse
from typing import Dict
from file2data import load_json, save_json


def merge_annotations_on_file(
    coco_file1: str, coco_file2: str, output_file: str
) -> None:
    """
    Merge annotations from two COCO JSON files.

    Args:
        coco_file1 (str): Path to the first COCO JSON file.
        coco_file2 (str): Path to the second COCO JSON file.
        output_file (str): Path to the output merged COCO JSON file.
    """
    coco1 = load_json(coco_file1)
    coco2 = load_json(coco_file2)

    merged_coco = coco1
    merged_coco["annotations"].extend(coco2["annotations"])

    save_json(output_file, merged_coco)


def merge_annotations_on_data(coco_data1: Dict, coco_data2: Dict) -> Dict:
    """
    Merge annotations from two COCO dictionaries.

    Args:
        coco_data1 (Dict): The first COCO dictionary.
        coco_data2 (Dict): The second COCO dictionary.

    Returns:
        Dict: The merged COCO dictionary.
    """
    merged_coco = coco_data1
    merged_coco["annotations"].extend(coco_data2["annotations"])
    return merged_coco


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge annotations from two COCO JSON files."
    )
    parser.add_argument("coco_file1", type=str, help="Path to the first COCO JSON file")
    parser.add_argument(
        "coco_file2", type=str, help="Path to the second COCO JSON file"
    )
    parser.add_argument(
        "output_file", type=str, help="Path to the output merged COCO JSON file"
    )
    args = parser.parse_args()

    merge_annotations_on_file(args.coco_file1, args.coco_file2, args.output_file)
