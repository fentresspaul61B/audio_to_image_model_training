import librosa
import numpy as np
import wave
import cv2


class Conf:
    """
    A set of settings that you can adapt to fit your audio files
    (frequency, average duration, number of Fourier, transforms...)
    """
    # Preprocessing settings
    sampling_rate = 44100
    duration = 2
    hop_length = 347 * duration  # to make time steps 128
    fmin = 20
    fmax = sampling_rate // 2
    n_mels = 128
    n_mfccs = 20
    n_fft = n_mels * 20
    samples = sampling_rate * duration


def is_valid_wav_file(filename):
    """
    Validate that a file is a valid WAV file.

    Parameters:
        filename (str): The name of the file to be validated.

    Returns:
        True if the file is a valid WAV file, False otherwise.
    """
    try:
        with wave.open(filename, 'rb') as wav_file:
            # Read the first chunk of the WAV file to check that it is a valid WAV file.
            wav_file.readframes(1)
            return True
    except wave.Error:
        return False
    pass


def read_audio(pathname, conf=Conf, trim_long_data=False):
    """
    Read in an audio file and return a NumPy array representing the audio data.

    Parameters:
        pathname (str): Path to the audio file to be read.
        conf (object): Configuration object with settings for audio processing.
        trim_long_data (bool): If True, trim audio data that is longer than the
        specified length in 'conf'. If False, pad shorter audio data with zeros
        to reach the specified length.

    Returns:
        audio_array (np.ndarray): NumPy array representing the audio data.

    Raises:
        Exception: If the audio file is not a valid .wav file.
    """

    # Validating that the incoming data is a .wav file. If not raises error.
    is_valid_wav_file(pathname)
    audio_array, sample_rate = librosa.load(pathname, sr=conf.sampling_rate)

    # trim silence, workaround: 0 length causes error, trim, top_db=default(60)
    if 0 < len(audio_array):
        audio_array, _ = librosa.effects.trim(audio_array)

    # make it unified length to conf.samples
    if len(audio_array) > conf.samples:
        if trim_long_data:
            audio_array = audio_array[0:0 + conf.samples]

    # If audio not long enough, add padding on both sides of array.
    else:
        padding = conf.samples - len(audio_array)
        offset = padding // 2
        audio_array = np.pad(audio_array,
                             (offset, conf.samples - len(audio_array) - offset),
                             'constant')
    return audio_array


def audio_array_to_mel_spectrogram_array(audio_array, conf=Conf):
    """
    Extract mel-scaled spectrogram features from audio data.

    Parameters:
        audio_array (np.ndarray): Audio data in the form of a 1-D numpy array.
        conf (object, optional): Configuration object containing relevant parameters
        for feature extraction. If not provided, the default configuration `conf`
        will be used.

    Returns:
        spectrogram (np.ndarray): Mel-scaled spectrogram of the input audio, with
        shape (n_mels, t).
    """
    spectrogram = librosa.feature.melspectrogram(
        y=audio_array,
        sr=conf.sampling_rate,
        n_mels=conf.n_mels,
        hop_length=conf.hop_length,
        n_fft=conf.n_fft,
        fmin=conf.fmin,
        fmax=conf.fmax)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram


def audio_array_to_mfcc_array(audio_array, conf=Conf):
    """
    Feature extraction function that takes in an audio array, and returns
    an array which is the MFCC of that audio array.
    Args:
        audio_array: numpy array of audio. 1D.
        conf: audio configurations used to read audio.
    Returns:
        mfcc: An array that represents the mfcc of the input.
    """
    mfcc = librosa.feature.mfcc(
        y=audio_array,
        sr=conf.sampling_rate,
        n_mfcc=Conf.n_mfccs,
        hop_length=Conf.hop_length,
        n_fft=Conf.n_fft,
        fmin=Conf.fmin,
        fmax=Conf.fmax)
    mfcc = librosa.power_to_db(mfcc)
    mfcc = mfcc.astype(np.float32)
    return mfcc


def audio_array_to_chroma_array(audio_array, conf=Conf):
    """
    Feature extraction function that takes in an audio array, and returns
    an array which is the chroma_stft of that audio array.
    Args:
        audio_array: numpy array of audio. 1D.
        conf: audio configurations used to read audio.
    Returns:
        chroma: An array that represents the chroma_stft of the input.
    """
    chroma = librosa.feature.chroma_stft(
        y=audio_array,
        sr=conf.sampling_rate,
        hop_length=Conf.hop_length,
        n_fft=Conf.n_fft)
    chroma = librosa.power_to_db(chroma)
    chroma = chroma.astype(np.float32)
    return chroma


def stretch_image_vertically(image_array, new_image_height):
    """
    This function is used to stretch_image_vertically in order to create
    square images for training the image network. Not all the images are
    stretched when creating the image dataset. This is used for MFCC and
    chroma.
    Args:
        image_array: np.array of image values.
        new_image_height: height dimension to stretch to. Currently,
        I use height of 50 to stretch the images, which is arbitrary.
    Returns:
        image_stretched_vertically: numpy array of the resized image.
    """
    if (type(new_image_height) != float) and (type(new_image_height) != int):
        raise ValueError("Value error: 'new_image_height' is not float or int.")

    if type(image_array) != np.ndarray:
        raise ValueError("Value error: 'image_array' is not np.ndarray.")

    if new_image_height <= 0:
        raise ValueError("The stretch factor must be greater than 0.")


    original_width = image_array.shape[1]
    stretched_image_dimensions = (original_width, new_image_height)
    image_stretched_vertically = cv2.resize(image_array,
                                            stretched_image_dimensions,
                                            interpolation=cv2.INTER_LINEAR)
    return image_stretched_vertically


def func1(num: float) -> float:
    """Multiply the num by 2. For testing purposes."""
    return num * 2


def func2(num: float) -> float:
    """Square the num. For testing purposes."""
    return num ** 2


def main():
    pass


if __name__ == "__main__":
    main()
