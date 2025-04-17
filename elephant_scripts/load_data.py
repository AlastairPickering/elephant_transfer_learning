"""
This script loads sets up the directories used to load the audio and the metadata dataframe while also reading in the metadata file and defining the label columns.
It then associates the audio files with their respective entries in the dataframe.
Script uses code from tutorial by https://github.com/marathomas/tutorial_repo#readme

"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

# Project directory. By default set to current working directory.
P_DIR = Path.cwd()

# Audio directory, contains audio (.wav) files.
AUDIO_IN = P_DIR / "audio_dir"

# Empty data directory, output files will be put here.
DATA = P_DIR / "data"

# Information about the info_file.csv
LABEL_COL = "call-type"  # -->name of column that contains labels

CALL_TYPE_PATH = DATA / "1.elephant_call_type_df.csv"

NA_DESCRIPTORS = [
    0,
    np.nan,
    "NA",
    "na",  # Which values indicate that this vocalisation is unlabelled
    "not available",
    "None",
    "Unknown",
    "unknown",
    None,
    "",
]


def load_vocalisation_dataset(
    path: Path | str = CALL_TYPE_PATH,
    audio_dir: Path | str = AUDIO_IN,
    data_dir: Path | str = DATA,
):
    path = Path(path)
    audio_dir = Path(audio_dir)
    data_dir = Path(data_dir)

    # Check if directories are present
    if not audio_dir.exists():
        raise FileNotFoundError("No audio directory found")

    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        print("Data directory created:", DATA)

    # Read in files

    if path.is_file():
        df = pd.read_csv(path)
        print("Info file loaded:", path)
    else:
        print("Input file missing:", path)
        print("Creating default input file without labels")

        audiofiles = os.listdir(audio_dir)

        if not audiofiles:
            raise FileNotFoundError("No audio files found in audio directory")

        df = pd.DataFrame(
            {
                "filename": [os.path.basename(x) for x in audiofiles],
                "label": ["unknown"] * len(audiofiles),
            }
        )
        print("Default input file created")

    audiofiles = df["filename"].values
    files_in_audio_directory = os.listdir(audio_dir)

    # Are there any files that are in the info_file.csv, but not in AUDIO_IN?
    missing_files = list(set(audiofiles) - set(files_in_audio_directory))
    if len(missing_files) > 0:
        print(
            "Warning:",
            len(missing_files),
            "files with no matching audio in audio folder",
        )

    df["audio_filepaths"] = [audio_dir / x for x in audiofiles]
    print("Audio file paths added to DataFrame")

    print("Vocalisation Dataset successfully loaded")

    return df
