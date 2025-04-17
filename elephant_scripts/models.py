import os

# Remove tensorflow messages about GPU usage
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from abc import ABC, abstractmethod
from pathlib import Path

import absl.logging
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Remove the warnings every time we load a model
absl.logging.set_verbosity(absl.logging.ERROR)


class Model(ABC):
    window_size: float
    samplerate: int

    @abstractmethod
    def extract_features(self, wav: np.ndarray) -> np.ndarray: ...


class VGGish(Model):
    window_size = 1
    samplerate = 16_000

    def __init__(self) -> None:
        self.model = hub.load("https://tfhub.dev/google/vggish/1")

    def extract_features(self, wav: np.ndarray) -> np.ndarray:
        if wav.ndim > 2:
            wav = wav.squeeze()

        embeddings = self.model(wav)  # type: ignore
        return embeddings.numpy()


class Perch(Model):
    window_size = 5
    samplerate = 32_000

    def __init__(self) -> None:
        self.model = hub.load(
            "https://www.kaggle.com/models/google/bird-vocalization-classifier/TensorFlow2/bird-vocalization-classifier/4"
        )

    def extract_features(self, wav: np.ndarray) -> np.ndarray:
        if wav.ndim == 1:
            wav = wav[np.newaxis, :]

        _, embeddings = self.model.infer_tf(wav)  # type: ignore
        return embeddings.numpy()


class YAMNet(Model):
    window_size = 1
    samplerate = 16_000

    def __init__(self) -> None:
        self.model = hub.load("https://tfhub.dev/google/yamnet/1")

    def extract_features(self, wav: np.ndarray) -> np.ndarray:
        if wav.ndim > 1:
            wav = wav.squeeze()

        _, embeddings, _ = self.model(wav)  # type: ignore
        return embeddings.numpy()


class BirdNET(Model):
    window_size = 3
    samplerate = 48_000

    def __init__(self) -> None:
        saved_model_dir = (
            Path(__file__).parent / "models" / "BirdNET_GLOBAL_6K_V2.4_Model"
        )

        model = tf.saved_model.load(saved_model_dir)
        self.infer = model.signatures["embeddings"]  # type: ignore

    def extract_features(self, wav: np.ndarray) -> np.ndarray:
        if wav.ndim == 1:
            wav = wav[np.newaxis, :]

        inputs = tf.convert_to_tensor(wav, dtype=tf.float32)
        outputs = self.infer(inputs=inputs)  # type: ignore
        return outputs["embeddings"].numpy()
