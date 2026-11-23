"""Tests for WavLM manifest handling, chunking, and row-aligned output."""

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from feature_extraction.extract_voice_features_wavlm import (
    content_hash,
    load_pcm16_wav,
    read_manifest,
    split_chunks,
    write_outputs,
)


class WavLMExtractionTests(unittest.TestCase):
    def test_training_and_evaluation_manifest_formats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "train.txt"
            training.write_text("p1 1 voices/a.wav faces/a.jpg id001 7\n")
            evaluation = root / "eval.txt"
            evaluation.write_text("p2 test/voices/b.wav test/faces/b.jpg\n")
            train_rows, is_training = read_manifest(training)
            eval_rows, is_evaluation_training = read_manifest(evaluation)
            self.assertTrue(is_training)
            self.assertFalse(is_evaluation_training)
            self.assertEqual(train_rows[0].class_index, 7)
            self.assertEqual(eval_rows[0].voice_path, "test/voices/b.wav")

    def test_chunking_covers_tail_deterministically(self) -> None:
        waveform = np.arange(25, dtype=np.float32)
        chunks = split_chunks(waveform, chunk_samples=10, overlap_samples=5)
        self.assertEqual(len(chunks), 4)
        np.testing.assert_array_equal(chunks[0], waveform[:10])
        np.testing.assert_array_equal(chunks[-1], waveform[-10:])

    def test_pcm_loading_and_hashing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.wav"
            samples = np.asarray([-32768, 0, 32767], dtype="<i2")
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(samples.tobytes())
            loaded = load_pcm16_wav(path)
            self.assertEqual(loaded.shape, (3,))
            self.assertEqual(len(content_hash(path)), 64)

    def test_duplicate_hashes_preserve_manifest_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "train.txt"
            manifest.write_text(
                "p1 1 voices/a.wav faces/a.jpg id001 7\n"
                "p2 1 voices/b.wav faces/b.jpg id001 7\n"
            )
            rows, is_training = read_manifest(manifest)
            vector = np.zeros(512, dtype=np.float32)
            vector[0] = 1.0
            matrix = write_outputs(
                rows,
                ["same", "same"],
                {"same": vector},
                root / "features.csv",
                root / "rowmap.csv",
                is_training,
            )
            self.assertEqual(matrix.shape, (2, 512))
            np.testing.assert_array_equal(matrix[0], matrix[1])
            saved = np.loadtxt(root / "features.csv", delimiter=",")
            self.assertEqual(saved.shape, (2, 513))
            np.testing.assert_array_equal(saved[:, -1], [7, 7])


if __name__ == "__main__":
    unittest.main()
