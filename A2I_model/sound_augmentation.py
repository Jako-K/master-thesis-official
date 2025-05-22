"""
Basic sound augmentation functions.

Includes cropping, noise addition, and volume scaling
for raw audio waveforms.
"""

import simple_asserts as S
import numpy as np
import random

def center_crop(waveform:np.ndarray, expected_sample_length=48_000 * 10) -> np.ndarray:
    """
    Crop the waveform to the center or pad it if it's too short.

    :param waveform: 1D numpy array of audio samples
    :param expected_sample_length: Target length after cropping or padding
    :return: Cropped or padded waveform of shape (expected_sample_length,)
    """
    # Input checks
    S.assert_type(waveform, np.ndarray, "waveform")
    S.assert_positive_int(waveform.shape[0])
    S.assert_positive_int(expected_sample_length)

    # Center crop
    if waveform.shape[0] >= expected_sample_length:
        start_sample = (waveform.shape[0] - expected_sample_length) // 2
        end_sample = start_sample + expected_sample_length
        waveform_cropped = waveform[start_sample:end_sample]
    else:
        # If the waveform is shorter than expected, pad it symmetrically
        padding_needed = expected_sample_length - waveform.shape[0]
        pad_left = padding_needed // 2
        pad_right = padding_needed - pad_left
        waveform_cropped = np.pad(waveform, (pad_left, pad_right), mode='constant')
    return waveform_cropped

def random_crop(waveform:np.ndarray, expected_sample_length=48_000 * 10) -> np.ndarray:
    """
    Crop a random segment of the waveform.

    :param waveform: 1D numpy array of audio samples
    :param expected_sample_length: Length of the cropped segment
    :return: Randomly cropped waveform segment
    """
    # Input checks
    S.assert_type(waveform, np.ndarray, "waveform")
    S.assert_positive_int(waveform.shape[0])
    S.assert_positive_int(expected_sample_length)

    # Random crop
    latest_starting_sample = waveform.shape[0] - expected_sample_length
    start_sample = random.randint(0, latest_starting_sample)
    end_sample = start_sample + expected_sample_length
    waveform_cropped = waveform[start_sample:end_sample]
    return waveform_cropped

def add_noise(waveform:np.ndarray, noise_level=0.005) -> np.ndarray:
    """
    Add random Gaussian noise to the waveform.

    :param waveform: 1D numpy array of audio samples
    :param noise_level: Scaling factor for the noise
    :return: Noisy waveform
    """
    # Input checks
    S.assert_type(waveform, np.ndarray, "waveform")
    S.assert_positive_int(waveform.shape[0])
    S.assert_positive_float(noise_level)

    noise = np.random.randn(len(waveform))
    augmented_waveform = waveform + noise_level * noise
    return augmented_waveform

def random_volume_scale(waveform:np.ndarray, min_scale=0.8, max_scale=1.2) -> np.ndarray:
    """
    Scale the waveform volume by a random factor.

    :param waveform: 1D numpy array of audio samples
    :param min_scale: Minimum scaling factor
    :param max_scale: Maximum scaling factor
    :return: Scaled waveform
    """
    # Input checks
    S.assert_type(waveform, np.ndarray, "waveform")
    S.assert_positive_int(waveform.shape[0])
    S.assert_positive_float(min_scale)
    S.assert_positive_float(max_scale, max_value_allowed=10.0) # TODO: Test what max value is reasonable.

    scale = random.uniform(min_scale, max_scale)
    return waveform * scale

##############################################
# Not used (too slow)
##############################################

# def time_stretch(waveform, rate=1.0):
#     return librosa.effects.time_stretch(waveform, rate=rate)
#
# def pitch_shift(waveform, sr=48000, n_steps=0):
#     return librosa.effects.pitch_shift(waveform, sr=sr, n_steps=n_steps)
#
# # Augmenting the fixed-size cropped waveform
# def augment_waveform(raw_waveform, sr=48000, expected_sample_length=48_000 * 10):
#     # Step 1: Crop to fixed size
#     waveform = random_crop(raw_waveform, expected_sample_length)
#
#     # Step 2: Apply random augmentations
#     if random.random() < 0.5:
#         rate = random.uniform(0.8, 1.0)
#         waveform = time_stretch(waveform, rate)
#
#     if random.random() < 0.5:
#         n_steps = random.randint(-2, 2)
#         waveform = pitch_shift(waveform, sr, n_steps)
#
#     if random.random() < 0.5:
#         waveform = add_noise(waveform)
#
#     if random.random() < 0.5:
#         waveform = random_volume_scale(waveform)
#
#     # Ensure it’s still the right length
#     waveform = waveform[:expected_sample_length]
#     assert waveform.shape[0] == expected_sample_length, waveform.shape
#     return waveform
