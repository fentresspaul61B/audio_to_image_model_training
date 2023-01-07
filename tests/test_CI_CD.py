from src.packaged_logic_for_CI_CD.main import func1, func2, read_audio  # Functions for testing.
import numpy as np  # Use for testing arrays.
import unittest  # Used for writing unit test.
import wave  # Used for testing valid wav files.

TEST_VALID_AUDIO_FILE = "tests/test_data/cat_audio_for_testing.wav"
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
