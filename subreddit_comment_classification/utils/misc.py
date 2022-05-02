"""
Miscellaneous utility functions.
"""

from pathlib import Path
from typing import Any, Dict


def full_path(relative_path: str) -> Path:
    return Path(relative_path).resolve()


def pretty_dumps(d: Dict[Any, Any]) -> None:
    delimiter = '", "'
    return f'["{delimiter.join(sorted(d))}"]'
