import librosa
import numpy as np
import wave


# A set of settings that you can adapt to fit your audio files (frequency, average duration, number of Fourier
# transforms...)
class conf:
    # Preprocessing settings
    sampling_rate = 44100
    duration = 2
    hop_length = 347*duration  # to make time steps 128
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
    - filename (str): The name of the file to be validated.

    Returns:
    - True if the file is a valid WAV file, False otherwise.
    """
    try:
        with wave.open(filename, 'rb') as wav_file:
            # Read the first chunk of the WAV file to check that it is a valid WAV file.
            wav_file.readframes(1)
    except wave.Error:
        return False

def read_audio(pathname, conf=conf, trim_long_data=False):
    """
    Read in an audio file and return a NumPy array representing the audio data.

    Parameters:
    - pathname (str): Path to the audio file to be read.
    - conf (object): Configuration object with settings for audio processing.
    - trim_long_data (bool): If True, trim audio data that is longer than the
      specified length in 'conf'. If False, pad shorter audio data with zeros
      to reach the specified length.

    Returns:
    - audio_array (np.ndarray): NumPy array representing the audio data.

    Raises:
    - Exception: If the audio file is not a valid .wav file.
    """

    # Validating that the incoming data is a .wav file. If not raises error.
    # is_valid_wav_file(pathname)

    audio_array, sample_rate = librosa.load(pathname, sr=conf.sampling_rate)

    # trim silence
    if 0 < len(audio_array):
        # workaround: 0 length causes error
        # trim, top_db=default(60)
        audio_array, _ = librosa.effects.trim(audio_array)

    # make it unified length to conf.samples
    if len(audio_array) > conf.samples:

        # long enough
        if trim_long_data:

            # adding audio data to array.
            audio_array = audio_array[0:0+conf.samples]

    # If audio not long enough, add padding on both sides of array.
    else:
        padding = conf.samples - len(audio_array)
        offset = padding // 2
        audio_array = np.pad(audio_array, (offset, conf.samples - len(audio_array) - offset), 'constant')

    return audio_array


def func1(num: float) -> float:
    """Multiply the num by 2."""
    return num * 2


def func2(num: float) -> float:
    """Square the num."""
    return num ** 2


def main():
    test_audio = read_audio("tests/test_data/invalid_data.wav")

    # print(test_audio == False)
    # print("Code finished.")
    pass


if __name__ == "__main__":
    main()
