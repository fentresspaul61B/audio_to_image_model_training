from src.packaged_logic_for_CI_CD.main import func1, func2, read_audio
from src.packaged_logic_for_CI_CD.main import audio_array_to_mel_spectrogram_array, audio_array_to_mfcc_array
from src.packaged_logic_for_CI_CD.main import audio_array_to_chroma_array
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


class BaseAudioRepresentationTest(unittest.TestCase):
    def check_representation_validity(self, audio_representation):
        """
        Check if the given audio representation is a valid numpy array and contains non-zero values.

        Parameters:
        audio_representation : np.ndarray
            The audio representation to be checked for validity.

        Raises:
            AssertionError: If `audio_representation` is not a numpy array, or if it contains all zeros,
            or if the ratio of non-zero values is less than 1%.
        """
        self.assertTrue(isinstance(audio_representation, np.ndarray))

        non_zero_values = [x for x in audio_representation if abs(x.all()) > 0]
        ratio_of_non_zero_values = len(non_zero_values) / len(audio_representation)

        self.assertTrue(len(non_zero_values) > 0, "Array contains all zeros")
        self.assertTrue(ratio_of_non_zero_values > 0.01, "Array contains some values")


class TestMelSpectrogram(BaseAudioRepresentationTest):
    def test_mel_valid_audio(self):
        audio_array = read_audio(TEST_VALID_AUDIO_FILE)
        mel_spectrogram = audio_array_to_mel_spectrogram_array(audio_array)
        self.check_representation_validity(mel_spectrogram)


class TestMFCC(BaseAudioRepresentationTest):
    def test_mfcc_valid_audio(self):
        audio_array = read_audio(TEST_VALID_AUDIO_FILE)
        mfcc = audio_array_to_mfcc_array(audio_array)
        self.check_representation_validity(mfcc)


class TestChroma(BaseAudioRepresentationTest):
    def test_chroma_valid_audio(self):
        audio_array = read_audio(TEST_VALID_AUDIO_FILE)
        chroma = audio_array_to_chroma_array(audio_array)
        self.check_representation_validity(chroma)

