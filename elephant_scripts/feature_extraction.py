"""
This script carries out all of the audio pre-processing steps before extracting
the acoustic features (embeddings) using the pre-trained CNN vggish and adding
in the missing duration information.
"""
# Import libraries

import warnings
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.axes import Axes

from .types import Annotation

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

# Set path

P_DIR = Path.cwd()

# Here we define the default values for the parameters that will be used in the
# audio pre-processing steps.
DEFAULT_SAMPLERATE = 4000  # Hz

# This is the duration of audio that will be extracted from the original audio
# and passed to the vggish model. The vggish model takes 0.96s
# of audio at 16kHz as input, which is 15360 samples. Since the samplerate of
# the audio files is 4000Hz, we need to change the window duration to 4s
# in order to extract the correct number of samples.
DEFAULT_WINDOW_SIZE = 4  # seconds

# Audio of annotations will be extracted from the original audio file and
# inserted into a 4s window. This parameter defines the position of the
# annotation within the 4s window.
DEFAULT_CLIP_POSITION = "middle"

# The default values for the parameters used in the spectrogram generation
# This is the number of samples in each window of the STFT.
DEFAULT_NFFT = 2048
# This parameter defines the number of samples between successive frames.
DEFAULT_HOP_LENGTH = 512

# Define functions needed to read audio files, apply frequency bandpass filter,
# extract the annotation based on timestamp, normalise amplitude, zero-pad and
# centre the annotation with the vggish-compatible 0.96s input window.


def read_audio(
    path: Path | str,
    samplerate: int | None = DEFAULT_SAMPLERATE,
) -> tuple[np.ndarray, int]:
    wav, sr = librosa.load(path, sr=samplerate)
    return wav, int(sr)


def apply_bandpass_filter(
    wav: np.ndarray,
    low_freq: float,
    high_freq: float,
    samplerate: int = DEFAULT_SAMPLERATE,
    order: int = 5,
) -> np.ndarray:
    # Define the bandpass filter
    sos = scipy.signal.butter(
        order,
        [low_freq, high_freq],
        fs=samplerate,
        btype="band",
        output="sos",
    )

    # Apply the bandpass filter to the signal
    return scipy.signal.sosfilt(sos, wav)


def extract_audio(
    wav: np.ndarray,
    annotation: Annotation,
    samplerate: int = DEFAULT_SAMPLERATE,
) -> np.ndarray:
    start_index = int(annotation.start_time * samplerate)
    end_index = int(annotation.end_time * samplerate)

    # Adjust start and end indices to ensure they are within the audio boundaries
    start_index = max(0, start_index)
    end_index = min(len(wav), end_index)

    extracted_audio = wav[start_index:end_index]
    return extracted_audio


def zero_pad(
    wav: np.ndarray,
    annotation: Annotation,
    window_size: float = DEFAULT_WINDOW_SIZE,
    samplerate: int = DEFAULT_SAMPLERATE,
    position: str = "middle",
) -> np.ndarray:
    num_windows = np.ceil((annotation.end_time - annotation.start_time) / window_size)
    return_size = int(num_windows * window_size * samplerate)
    padding_needed = max(0, return_size - len(wav))

    if position == "middle":
        start_index = max(0, padding_needed // 2)
    elif position == "start":
        start_index = 0
    elif position == "end":
        start_index = padding_needed
    else:
        raise ValueError("Position must be one of 'middle', 'start', or 'end'.")

    return np.pad(wav, (start_index, return_size - len(wav) - start_index))


def normalise_sound_file(wav: np.ndarray) -> np.ndarray:
    # Calculate the peak amplitude
    peak_amplitude = np.max(np.abs(wav))

    # Set the whole sound file to the peak amplitude
    normalised_data = wav * (1 / peak_amplitude)

    return normalised_data


def generate_spectrogram(
    wav: np.ndarray,
    samplerate: int = DEFAULT_SAMPLERATE,
    n_fft: int = DEFAULT_NFFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
):
    spectrogram = librosa.feature.melspectrogram(
        y=wav,
        sr=samplerate,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
    )
    return librosa.power_to_db(spectrogram, ref=np.max)


def wav_cookiecutter(
    annotation: Annotation,
    window_size: float = DEFAULT_WINDOW_SIZE,
    position: str = DEFAULT_CLIP_POSITION,
    samplerate: int = DEFAULT_SAMPLERATE,
):
    """Extract the acoustic features of a single annotation."""
    # Get path of the audio file from the annotation info
    path = str(annotation.audio_filepaths)

    # Read the audio
    wav, sr = read_audio(path)

    # If the samplerate of the audio file does not match the samplerate
    # then the filtering and annotation extraction will produce
    # incorrect results, so we need to check this.
    assert sr == samplerate

    # Apply the bandpass filter to the signal
    wav = apply_bandpass_filter(
        wav,
        annotation.low_freq,
        annotation.high_freq,
        samplerate=samplerate,
    )

    # Extract audio segment based on annotation times
    wav = extract_audio(wav, annotation, samplerate=samplerate)

    # Zero-pad the wav array based on the annotation
    wav = zero_pad(
        wav,
        annotation,
        window_size=window_size,
        samplerate=samplerate,
        position=position,
    )

    # Normalise the sound file
    wav = normalise_sound_file(wav)

    # Re-apply the bandpass filter to remove artifacts
    wav = apply_bandpass_filter(
        wav,
        annotation.low_freq,
        annotation.high_freq,
        samplerate=samplerate,
    )

    return wav


def plot_spectrogram(
    audio: np.ndarray,
    sr: int = DEFAULT_SAMPLERATE,
    ax: Axes | None = None,
    figsize: tuple[int, int] = (5, 5),
    cmap: str = "magma",
):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    img = librosa.display.specshow(
        log_spectrogram,
        sr=sr,
        x_axis="time",
        y_axis="mel",
        ax=ax,
        cmap=cmap,
    )

    return ax, img


def plot_spectrograms(
    steps,
    audios,
    sr=DEFAULT_SAMPLERATE,
    num_rows=3,
    num_cols=2,
    figsize=(16, 16),
    window_size=DEFAULT_WINDOW_SIZE,
    position=DEFAULT_CLIP_POSITION,
    fontsize: int | None = 20,
):
    """Function to plot spectrograms."""
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=figsize,
        sharey=True,
    )

    # Create an additional axis for the color bar
    # Adjust [left, bottom, width, height]
    cbar_ax = fig.add_axes((0.92, 0.15, 0.03, 0.7))

    labels = ["a)", "b)", "c)", "d)", "e)", "f)"]

    for i, (step, audio) in enumerate(zip(steps, audios)):
        row = i // num_cols
        col = i % num_cols

        ax = axes[row, col]

        spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
        log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        img = librosa.display.specshow(
            log_spectrogram,
            sr=sr,
            x_axis="time",
            y_axis="mel",
            ax=ax,
            cmap="magma",
        )

        # Adjust font sizes of x and y-axis labels and tick labels
        # Set font size for x-axis label
        ax.set_xlabel("Time (s)", fontsize=fontsize)

        # Set font size for y-axis label
        ax.set_ylabel("Frequency (Hz)", fontsize=fontsize)

        # Set font size for tick labels
        ax.tick_params(axis="both", which="both", labelsize=fontsize)

        # Add subplot label
        ax.text(-0.1, 1.0, labels[i], transform=ax.transAxes, size=fontsize)

    # Add a single colour bar for the entire figure
    cbar = fig.colorbar(img, cax=cbar_ax, format="%+2.0f dB")
    cbar.ax.tick_params(labelsize=fontsize)

    # plt.tight_layout(rect=(0, 0, 0.9, 1))  # Adjust layout to make room for colour bar

    # Increase space between subplots
    plt.subplots_adjust(hspace=0.4)

    plt.show()
