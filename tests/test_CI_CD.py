from src.packaged_logic_for_CI_CD.main import func1, func2, read_audio
from src.packaged_logic_for_CI_CD.main import audio_array_to_mel_spectrogram_array, audio_array_to_mfcc_array
from src.packaged_logic_for_CI_CD.main import audio_array_to_chroma_array
from src.packaged_logic_for_CI_CD.main import stretch_image_vertically
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


def create_dummy_image(width, height, channels):
    # create a numpy array with all zeros of the specified shape
    img = np.zeros((height, width, channels), dtype=np.uint8)
    return img


class TestStretchImage(unittest.TestCase):
    def test_stretching(self):
        """
        This test to checks the dimensions of the output image and
        verify that it matches the new_image_height argument and the
        original width of the input image for each audio feature.
        """
        desired_height = 50

        # Reading in audio file.
        test_audio = read_audio(TEST_VALID_AUDIO_FILE)

        # Creating arrays for each feature type.
        audio_functions = [audio_array_to_mfcc_array,
                           audio_array_to_mel_spectrogram_array,
                           audio_array_to_chroma_array]

        # computing the desired, and actual stretched shapes.
        audio_arrays = [f(test_audio) for f in audio_functions]
        audio_widths = [arr.shape[1] for arr in audio_arrays]
        desired_shapes = [(desired_height, width) for width in audio_widths]
        stretched_audio = [stretch_image_vertically(arr, desired_height) for arr in audio_arrays]
        actual_shapes = [stretched.shape for stretched in stretched_audio]

        # Validating that the desired and actual are equal.
        for desired_shape, actual_shape in zip(desired_shapes, actual_shapes):
            self.assertEqual(desired_shape, actual_shape)

    def test_stretch_accuracy(self):
        """
        Check for stretching accuracy: Write a test to check that the
        stretching is done accurately, meaning that the aspect ratio of
        the image is preserved. This can be done by creating an image of
        a known aspect ratio and checking that the aspect ratio of the output
        image is the same.
        """
        # Create a dummy image of known aspect ratio
        dummy_image = create_dummy_image(25, 50, 3)

        # Stretch the dummy image vertically
        stretched_image = stretch_image_vertically(dummy_image, 50)

        # Check that the height of the stretched image is equal to the desired height
        self.assertEqual(stretched_image.shape[0], 50)

        # Check that the aspect ratio of the stretched image is equal to the original aspect ratio
        original_aspect_ratio = dummy_image.shape[1] / dummy_image.shape[0]
        stretched_aspect_ratio = stretched_image.shape[1] / stretched_image.shape[0]
        self.assertAlmostEqual(original_aspect_ratio, stretched_aspect_ratio, delta=1e-6)

    def test_stretch_edge_cases(self):
        """
        Verify that the function can handle edge cases, such as an input
        image of height 1, an input image with a width of 1, and a
        new_image_height of 1.
        """
        tiny_image = create_dummy_image(1, 1, 3)
        stretched_image = stretch_image_vertically(tiny_image, 50)
        self.assertTrue(type(stretched_image) == np.ndarray)

    def test_stretch_negative_values(self):
        """
        Check for negative values: Write a test to check that the function
        raises an error when a negative value is passed as the new_image_height
        argument.
        """
        dummy_image = create_dummy_image(1, 1, 3)

        with self.assertRaises(ValueError, msg="Stretch factor must be greater than 0"):
            stretch_image_vertically(dummy_image, -50)

        dummy_image = create_dummy_image(224, 224, 3)

        with self.assertRaises(ValueError, msg="Stretch factor must be greater than 0"):
            stretch_image_vertically(dummy_image, -0.5)

    def test_stretch_non_integer_values(self):
        """
        Check for non-integer values: Write a test to check that the function
        raises an error when a non-integer value is passed as the new_image_height
        argument.
        """

    def test_stretch_catches_invalid_inputs(self):
        """
        Check for invalid inputs: Write tests to verify that the function raises
        an error when an invalid input image is provided, such as a 1D array instead
        of a 2D array.
        :return:
        """
