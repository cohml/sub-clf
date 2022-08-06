"""
Miscellaneous utility functions.
"""

from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict


def full_path(relative_path: str) -> Path:
    return Path(relative_path).resolve()


def measure_duration(f: Callable) -> Callable:
    """Decorator to measure function execution time. For use with `collect/tally.py`"""

    def measured_function(*args) -> None:
        start = perf_counter()
        result = f(*args)
        end = perf_counter()

        secs = round(end - start)
        mins = int(secs / 60)
        if mins > 0:
            duration = f'{mins}m {secs % mins}s'
        else:
            duration = f'{secs}s'
        print(f'done ({duration})')

        return result

    return measured_function


def pretty_dumps(d: Dict[str, Any]) -> str:
    delimiter = '", "'
    return f'["{delimiter.join(sorted(d))}"]'
