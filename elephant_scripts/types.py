from pathlib import Path
from typing import Protocol


class Annotation(Protocol):
    start_time: float
    end_time: float
    low_freq: float
    high_freq: float
    audio_filepaths: Path
