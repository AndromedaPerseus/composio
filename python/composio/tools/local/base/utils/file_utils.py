import os
from pathlib import Path


def get_rel_fname(root, fname):
    """Get relative file name from the root directory."""
    return os.path.relpath(fname, root)


def split_path(root, path):
    """Split path into components relative to the root directory."""
    path = os.path.relpath(path, root)
    return [path + ":"]


def get_mtime(fname):
    """Get modification time of a file."""
    try:
        return os.path.getmtime(fname)
    except FileNotFoundError:
        print(f"File not found error: {fname}")
        return None
