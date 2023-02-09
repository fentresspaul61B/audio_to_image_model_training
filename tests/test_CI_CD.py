from src.packaged_logic_for_CI_CD.main import func1, func2, read_audio
from src.packaged_logic_for_CI_CD.main import audio_array_to_melspectrogram_array, audio_array_to_MFCC
import numpy as np  # Use for testing arrays.
import unittest  # Used for writing unit test.
TEST_VALID_AUDIO_FILE = "tests/test_data/valid_data.wav"
TEST_INVALID_AUDIO_FILE = "tests/test_data/invalid_data.wav"


class TestMyFunction(unittest.TestCase):
    def test_func1(self):
        """Test for func1."""
        self.assertEqual(func1(2), 4)

    def test_func2(self):
        """Test for func2."""
        self.assertEqual(func2(2), 4)


class TestReadAudio(unittest.TestCase):
    def test_valid_audio(self):
        """
        Test that a valid audio file is correctly identified as valid.

        This function reads an audio file using the 'read_audio' function and
        checks that it is a valid NumPy array with non-zero values.
        """
        audio_array = read_audio(TEST_VALID_AUDIO_FILE)
        self.assertTrue(isinstance(audio_array, np.ndarray))
        self.assertTrue(audio_array.size != 0)
        self.assertTrue(not np.all(audio_array == 0))

    def test_invalid_audio(self):
        """
        Test that an invalid audio file is correctly identified as invalid.

        This function reads an audio file using the 'read_audio' function and
        checks that it is not a valid NumPy array.
        """
        with self.assertRaises(EOFError):
            audio_array = read_audio(TEST_INVALID_AUDIO_FILE)
            self.assertFalse(isinstance(audio_array, np.ndarray))
    pass


class TestMelSpectrogram(unittest.TestCase):
    def test_mel_valid_audio(self):
        audio_array = read_audio(TEST_VALID_AUDIO_FILE)
        mel_spectrogram = audio_array_to_melspectrogram_array(audio_array)
        self.assertTrue(isinstance(mel_spectrogram, np.ndarray))

        non_zero_values = [x for x in mel_spectrogram if abs(x.all()) > 0]
        ratio_of_non_zero_values = len(non_zero_values) / len(mel_spectrogram)

        self.assertTrue(ratio_of_non_zero_values > 0.5, "Array contains mostly zeros")
        self.assertTrue(len(non_zero_values) > 0, "Array contains all zeros")


class TestMFCC(unittest.TestCase):
    def test_mfcc_valid_audio(self):
        """
        Test that audio array to MFCC conversion returns a valid numpy array
        and that it is not mostly or all zeros. This test reads an audio file
        and converts it to a Mel-frequency cepstral coefficients (MFCC)
        representation. It then checks that the resulting MFCC representation
        is a numpy array, and that it is not mostly or all zeros.
        """
        audio_array = read_audio(TEST_VALID_AUDIO_FILE)
        mfcc = audio_array_to_MFCC(audio_array)
        self.assertTrue(isinstance(mfcc, np.ndarray))

        non_zero_values = [x for x in mfcc if abs(x.all()) > 0]
        ratio_of_non_zero_values = len(non_zero_values) / len(mfcc)

        self.assertTrue(ratio_of_non_zero_values > 0.5, "Array contains mostly zeros")
        self.assertTrue(len(non_zero_values) > 0, "Array contains all zeros")
