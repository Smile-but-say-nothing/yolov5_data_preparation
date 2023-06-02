import os


def get_file_name(path: str):
    return os.path.split(path)[1]


def get_file_prefix(path: str):
    return os.path.splitext(os.path.split(path)[1])[0]


def get_file_format(path: str):
    return os.path.splitext(os.path.split(path)[1])[1]
