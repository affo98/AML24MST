"""
Inspiration from: https://medium.com/analytics-vidhya/adding-noise-to-audio-clips-5d8cee24ccb8
"""

import os
import shutil
import numpy as np
import soundfile as sf
import numpy as np
import librosa
import warnings

from helper import genre_names

warnings.filterwarnings("ignore")


def get_white_noise(signal, SNR):
    """
    SNR in dB
    Given a signal and desired SNR, this gives the required Additive White Gaussian Noise (AWGN) what should be added to the signal to get the desired SNR.

    Run using: python create_noise_data.py
    """
    # RMS value of signal
    RMS_s = np.sqrt(np.mean(signal**2))
    # RMS values of noise
    RMS_n = np.sqrt(RMS_s**2 / (pow(10, SNR / 10)))
    # Additive white gausian noise. Thereore mean=0
    # Because sample length is large (typically > 40000)
    # we can use the population formula for standard daviation.
    # because mean=0 STD=RMS
    STD_n = RMS_n
    # generate noise with mean 0 and std=RMS_n
    noise = np.random.normal(0, STD_n, signal.shape[0])
    return noise


def create_train_test_data_noisy(SNR):
    """
    SNR defines signal to noise ratio. Lower SNR gives more noise.
    Requires a folder called data_train_test/. Run create_train_test_data.py to create that folder.
    Saves ONLY train dataset with injected noise according to SNR in the folder data_noisy_train_test.

    Run using: python create_noise_data.py
    """

    # Create directory for train
    data_noisy_train_path = "data_noisy_train"
    train_path = os.path.join(data_noisy_train_path, "train")

    # remove folder if it already exists
    if os.path.exists(data_noisy_train_path):
        shutil.rmtree(data_noisy_train_path)

    # Create the directories
    os.makedirs(train_path, exist_ok=True)

    # Create subdirectories inside train_path for each genre
    for genre_name in genre_names:
        genre_dir = os.path.join(train_path, genre_name)
        os.makedirs(genre_dir, exist_ok=True)

    print(f"Directory {data_noisy_train_path} created successfully.")

    # Looping over files in folder data_train_test/
    path = "./data_train_test/"
    # for set in ["train", "test"] - use if you want noisy test data aswell
    for set in ["train"]:
        path_set = os.path.join(path, set)
        for genre_name in genre_names:
            # Construct full path to genre directory
            genre_dir = os.path.join(path_set, genre_name)

            # Check if the directory exists
            if not os.path.isdir(genre_dir):
                print(f"Directory '{genre_dir}' does not exist.")
                continue

            # Loop over files in the genre directory
            for file_name in os.listdir(genre_dir):
                # Construct full path to file
                file_path = os.path.join(genre_dir, file_name)
                file_save_path = os.path.join(
                    data_noisy_train_path, set, genre_name, file_name
                )

                # Check if it's a file
                if os.path.isfile(file_path):
                    try:
                        signal, sr = librosa.load(
                            file_path
                        )  # load file into signal and sampling_rate
                        noise = get_white_noise(signal, SNR)  # generate noise,
                        signal_noise = signal + noise
                        sf.write(file_save_path, signal_noise, sr)
                    except Exception:
                        print(f"Could not load {file_name}")
                        continue

                else:
                    print(f"'{file_path}' is not a file.")
    print(
        f"Succesfully created train/test noisy data in folder {data_noisy_train_path}"
    )


__SNR__ = 10  # changed 29 april by Anders
if __name__ == "__main__":
    create_train_test_data_noisy(__SNR__)