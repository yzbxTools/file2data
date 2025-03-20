import orjson


def read_json(file_path):
    with open(file_path, "rb") as f:
        return orjson.loads(f.read())


def write_json(file_path, data):
    with open(file_path, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))


def load_json(file_path):
    with open(file_path, "rb") as f:
        return orjson.loads(f.read())


def save_json(file_path, data):
    with open(file_path, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
