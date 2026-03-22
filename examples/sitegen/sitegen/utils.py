"""Utility functions for static site generator."""

import shutil
from pathlib import Path


def copy_static_files(input_dir: Path, output_dir: Path) -> None:
    """
    Copy static files from input directory to output.
    
    Args:
        input_dir: Source directory containing content
        output_dir: Destination directory for output
    """
    static_dir = input_dir / "static"
    dest_static = output_dir / "static"
    
    if static_dir.exists() and static_dir.is_dir():
        if dest_static.exists():
            shutil.rmtree(dest_static)
        shutil.copytree(static_dir, dest_static)


def ensure_dir(path: Path) -> None:
    """
    Ensure a directory exists, create if needed.
    
    Args:
        path: Directory path to ensure
    """
    if not path.exists():
        path.mkdir(parents=True)
