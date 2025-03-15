import json
import configparser
import csv
import pandas as pd
import argparse
import os
import numpy as np
import jsonlines
from loguru import logger
import yaml
from typing import Any, List, Union


def load_file(file_path: str) -> Union[List[Any], pd.DataFrame]:
    """
    Load data from a file based on its extension.

    Args:
        file_path (str): Path to the file to load.

    Returns:
        Union[List[Any], pd.DataFrame]: Loaded data.
    """
    file_suffix = file_path.split(".")[-1].lower()
    try:
        if file_suffix == "ini":
            config = configparser.ConfigParser()
            config.read(file_path)
            data = []
            for section in config.sections():
                data.append(dict(config[section]))
            return data

        elif file_suffix == "json":
            with open(file_path, "r") as file:
                data = json.load(file)
            return data

        elif file_suffix == "txt":
            with open(file_path, "r") as file:
                data = [l.strip() for l in file.readlines()]
            return data

        elif file_suffix == "csv":
            with open(file_path, "r") as file:
                reader = csv.reader(file)
                data = list(reader)
            return data

        elif file_suffix == "jsonl":
            with open(file_path, "r") as file:
                data = [json.loads(line) for line in file]
            return data

        elif file_suffix == "xlsx":
            data = pd.read_excel(file_path)
            return data

        elif file_suffix == 'yaml':
            with open(file_path, 'r') as file:
                data = yaml.safe_load(file)
            return data

        else:
            raise ValueError(f"Unsupported file format: {file_suffix}")
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        raise


def save_file(file_path: str, data: Any) -> None:
    """
    Save data to a file based on its extension.

    Args:
        file_path (str): Path to the file to save.
        data (Any): Data to save.
    """
    file_suffix = file_path.split(".")[-1].lower()
    try:
        if file_suffix == "ini":
            config = configparser.ConfigParser()
            config_dict = {}
            for i, item in enumerate(data):
                config_dict[f"video{i}"] = item
            config.read_dict(config_dict)
            with open(file_path, "w") as file:
                config.write(file)

        elif file_suffix == "json":
            with open(file_path, "w") as file:
                json.dump(data, file)

        elif file_suffix == "txt":
            with open(file_path, "w") as file:
                for d in data:
                    file.write(f"{d}\n")

        elif file_suffix == "csv":
            with open(file_path, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(data)

        elif file_suffix == "jsonl":
            with open(file_path, "w") as file:
                for item in data:
                    file.write(json.dumps(item) + "\n")

        elif file_suffix == "xlsx":
            data.to_excel(file_path, index=False)

        elif file_suffix == 'yaml':
            with open(file_path, 'w') as file:
                yaml.dump(data, file)

        else:
            raise ValueError(f"Unsupported file format: {file_suffix}")
    except Exception as e:
        logger.error(f"Error saving file {file_path}: {e}")
        raise


def resume_writer(file_path: str, flush: bool = True) -> (List[Any], jsonlines.Writer):
    """
    Resume writing to a JSONL file.

    Args:
        file_path (str): Path to the JSONL file.
        flush (bool): Whether to flush the writer after each write.

    Returns:
        (List[Any], jsonlines.Writer): Existing data and the writer object.
    """
    try:
        if os.path.exists(file_path):
            data = load_file(file_path)
            writer = jsonlines.open(file_path, mode="a", flush=flush)
        else:
            data = []
            writer = jsonlines.open(file_path, mode="w", flush=flush)

        return data, writer
    except Exception as e:
        logger.error(f"Error resuming writer for file {file_path}: {e}")
        raise


def main() -> None:
    """
    Main function to test load_file function with different file paths.
    """
    parser = argparse.ArgumentParser(
        description="Test load_file function with different file paths."
    )
    parser.add_argument("file_path", type=str, help="Path to the file to load")
    args = parser.parse_args()

    file_path = args.file_path
    try:
        data = load_file(file_path)
        print(type(data))
        print(data[0:3])
    except Exception as e:
        logger.error(f"Error in main function: {e}")


if __name__ == "__main__":
    main()
