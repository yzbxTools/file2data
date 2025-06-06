import orjson
import os
import os.path as osp


def read_json(file_path: str) -> dict:
    with open(file_path, "rb") as f:
        return orjson.loads(f.read())


def write_json(file_path: str, data: dict) -> None:
    os.makedirs(osp.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))


def load_json(file_path: str) -> dict:
    with open(file_path, "rb") as f:
        return orjson.loads(f.read())


def save_json(file_path: str, data: dict) -> None:
    os.makedirs(osp.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
