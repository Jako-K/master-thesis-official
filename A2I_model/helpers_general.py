"""
General-purpose utilities used throughout the code base.
Include everything from video generation to simple asserts.
"""

import pandas as pd
from moviepy.editor import AudioClip, ImageClip
import numpy as np
from datetime import datetime
import psutil
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib import gridspec
import subprocess
import os
import simple_asserts as S
import warnings
import librosa
import random
from typing import List
import shutil
import tempfile

#####################################################
# Load audio
#####################################################

def load_song_whole_in_clips(path_mp3:str, sample_rate:int=48_000, expected_sample_length:int=48_000*10) -> List[np.ndarray]:
    """
    Load a full song and split it into non-overlapping clips of fixed length.

    NOTE: Left over samples will be discarded.
    For instance, `expected_sample_length=10` and `len(path_mp3) == 25` two clips
    will be generated from samples 0-10 and 10-20 while samples from 20-25 will be ignored.

    EXAMPLE:
    >> SONG_PATH = "../dataset/example_song.mp3" # 42 duration song clip
    >> sequence_of_song_clips = load_song_whole_in_clips(SONG_PATH)
    >> print([c.shape for c in sequence_of_song_clips])
    [(480000,), (480000,), (480000,), (480000,)]

    :param path_mp3: Path to .mp3 file (other formats might work but are not tested)
    :param sample_rate: Target sample rate for loading the audio (48000 is a safe choice)
    :param expected_sample_length: Number of samples per clip (e.g. 10 sec = 480000 at 48 kHz)
    :return: List of waveform clips (each a 1D np.ndarray of shape (expected_sample_length,))
    """

    # Input Checks
    S.assert_positive_int(sample_rate, zero_allowed=False, variable_name="sample_rate")
    S.assert_positive_int(expected_sample_length, zero_allowed=False, variable_name="expected_sample_length")
    S.assert_path_exists(path_mp3, "path_mp3")
    if not path_mp3.lower().endswith(".mp3"):
        warnings.warn("I have only tested the code with .mp3 files. It may well work with e.g. wav-files, but perhaps not..")

    raw_waveform, sr = librosa.load(path_mp3, sr=sample_rate)
    length = raw_waveform.shape[0]
    num_clips = length // expected_sample_length
    clips = []
    for clip_index in range(num_clips):
        start_index = clip_index*expected_sample_length
        end_index = (clip_index+1)*expected_sample_length
        raw_waveform_clip = raw_waveform[start_index:end_index]
        assert raw_waveform_clip.shape[0] == expected_sample_length
        clips.append(raw_waveform_clip)
    return clips


def load_song_clip(path_mp3:str, crop_method:str="center", sample_rate:int=48_000, expected_sample_length:int=48_000*10) -> np.ndarray:
    """
    Load a song and return a single clip (either center-cropped or randomly cropped).

    # EXAMPLE:
    >> SONG_PATH = "../dataset/example_song.mp3" # 42 duration song clip
    >> sequence_of_song_clips = load_song_clip(SONG_PATH, crop_method="center")
    >> print(a_single_song_clip.shape)
    (480000,)

    :param path_mp3: Path to .mp3 file (other formats might work but are not tested)
    :param crop_method: "center" or "random" define how the clip is extracted from the song
    :param sample_rate: Target sample rate for loading the audio (48000 is a safe choice)
    :param expected_sample_length: Number of samples in the output clip (e.g. 10 sec = 480000 at 48 kHz)
    :return: A 1D np.ndarray with shape (expected_sample_length,)
    """

    # Input Checks
    S.assert_type(crop_method, str, "crop_method")
    assert crop_method in ["center", "random"], f"Expected to `{crop_method=}` to be in `['center', 'random']`"
    S.assert_positive_int(sample_rate, zero_allowed=False, variable_name="sample_rate")
    S.assert_positive_int(expected_sample_length, zero_allowed=False, variable_name="expected_sample_length")
    S.assert_path_exists(path_mp3, "path_mp3")
    if not path_mp3.lower().endswith(".mp3"):
        warnings.warn("I have only tested the code with .mp3 files. It may well work with e.g. wav-files, but perhaps not..")

    raw_waveform, sr = librosa.load(path_mp3, sr=sample_rate)
    length = raw_waveform.shape[0]

    if crop_method == "center":
        assert length >= expected_sample_length, f"{length} >= {expected_sample_length}"
        center = int(length / 2)
        start_index = int(np.floor(center - expected_sample_length/2))
        end_index =   int(np.ceil(center + expected_sample_length/2))
        raw_waveform = raw_waveform[start_index:end_index]
        assert raw_waveform.shape[0] == (end_index-start_index) == expected_sample_length
    elif crop_method == "random":
        latest_starting_sample = length - expected_sample_length
        start_sample = random.randint(0, latest_starting_sample)
        end_sample = start_sample + expected_sample_length
        raw_waveform = raw_waveform[start_sample:end_sample]
    return raw_waveform


#####################################################
# Generate images and plots
#####################################################

def create_image_with_prompt(prompt: str, font_size: int = 50, text_start_height: int = 0) -> Image.Image:
    """
    Create a 1024x1024 white image and draw the prompt text onto it.

    Automatically wraps long lines to avoid overflowing the image width.

    # EXAMPLE:
    >> img = create_image_with_prompt("This is a sample\nPrompt text that wraps")
    >> img.show()

    :param prompt: Text string to draw onto the image (can include line breaks)
    :param font_size: Font size of the text (default: 50)
    :param text_start_height: Vertical pixel offset where the text starts (default: 0)
    :return: PIL Image with the rendered text
    """
    # Input checks
    S.assert_type(prompt, str, "prompt")
    S.assert_positive_int(font_size, zero_allowed=False)
    S.assert_positive_int(text_start_height, zero_allowed=True)

    # Create white image
    img = Image.new('RGB', (1024, 1024), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Load font
    try:
        font = ImageFont.truetype("arial.ttf", size=font_size)
    except IOError:
        font = ImageFont.load_default()

    # Word wrap
    line_spacing = 60  # Space between lines
    lines = []
    for line in prompt.split('\n'):
        words = line.split(' ')
        line_accum = ""
        for word in words:
            test_line = f"{line_accum} {word}".strip()
            bbox = draw.textbbox((0, 0), test_line, font=font)
            text_width = bbox[2] - bbox[0]
            if text_width <= 900:
                line_accum = test_line
            else:
                lines.append(line_accum)
                line_accum = word
        lines.append(line_accum)

    # Draw lines
    y = text_start_height
    for line in lines:
        draw.text((50, y), line, font=font, fill=(0, 0, 0))
        y += line_spacing

    return img

def epoch_plot(df_stats: pd.DataFrame, path_logging_folder: str, epoch: int) -> None:
    """
    Plot training and validation loss curves up to the current epoch and save info/plot as .csv/.jpg.

    # EXAMPLE:
    >> epoch_plot(stats_df, "logs", epoch=4)

    :param df_stats: DataFrame containing at least "train_loss" and "valid_loss" columns
    :param path_logging_folder: Folder path where outputs (stats.csv, stats.jpg) are saved
    :param epoch: Current training epoch (0-indexed) to plot up to
    :return: None
    """
    # Input checks
    S.assert_type(df_stats, pd.DataFrame, "stats")
    S.assert_folder_exists(path_logging_folder, "path_logging_folder")
    if "train_loss" not in df_stats.columns or "valid_loss" not in df_stats.columns:
        raise ValueError("`df_stats` must contain 'train_loss' and 'valid_loss' columns.")
    S.assert_positive_int(epoch, zero_allowed=True, variable_name="epoch")
    if not (0 <= epoch < len(df_stats)):
        raise ValueError(f"`epoch` must be in the range [0, {len(df_stats) - 1}], but got {epoch}.")

    # Setup
    stats_plot = df_stats.iloc[:epoch + 1]
    stats_plot.to_csv(f"{path_logging_folder}/stats.csv", index=False)
    is_valid_rows = df_stats["valid_loss"].diff() != 0

    # Set up the figure and grid layout
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[4, 5])

    # Top left: Train loss
    ax_train = fig.add_subplot(gs[0, 0])
    stats_plot["train_loss"].plot(style="-o", ax=ax_train, label="train", color="#1f77b4")
    ax_train.legend(["train"])

    # Top right: Valid loss
    ax_valid = fig.add_subplot(gs[0, 1])
    stats_plot["valid_loss"][is_valid_rows].plot(style="-o", ax=ax_valid, label="valid", color="#ff7f0e")
    ax_valid.legend(["valid"])

    # Bottom: Combined
    ax_combined = fig.add_subplot(gs[1, :])
    stats_plot["train_loss"].plot(style="-o", ax=ax_combined, label="train", color="#1f77b4")
    stats_plot["valid_loss"][is_valid_rows].plot(style="-o", ax=ax_combined, label="valid", color="#ff7f0e")
    ax_combined.legend(["train", "valid"])

    # Wrap up
    plt.tight_layout()
    plt.savefig(f"{path_logging_folder}/stats.jpg")
    plt.close(fig)

#####################################################
# Video
#####################################################

def combine_videos_ffmpeg(video_paths:List[str], video_output_path:str):
    """
    Combine multiple .mp4 files into a single video using ffmpeg.

    # EXAMPLE:
    >> combine_videos_ffmpeg(["a.mp4", "b.mp4", "c.mp4"], "combined.mp4")

    :param video_paths: List of paths to .mp4 video files (in order)
    :param video_output_path: .mp4 save path to of the video - (other formats might work but are not tested)
    :return: None
    """
    # Checks
    S.assert_type(video_paths, list, "video_paths")
    for video_path in video_paths:
        S.assert_type(video_path, str, "video_path")
    if any((not v.lower().endswith(".mp4")) for v in video_paths):
        warnings.warn("I have only tested the code with .mp4 files. It may well work with e.g. AVI-files, but perhaps not..")

    S.assert_path_dont_exists(video_output_path, "video_output_path")
    if not video_output_path.lower().endswith(".mp4"):
        warnings.warn("I have only tested the code with .mp4 files. It may well work with e.g. AVI-files, but perhaps not..")

    # Create a temporary text file listing all video files
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        txt_path = temp_file.name
    with open(txt_path, "w") as f:
        for video_path in video_paths:
            f.write(f"file '{video_path}'\n")

    # Run ffmpeg to concatenate videos
    ffmpeg_command = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", txt_path,
        "-c", "copy",
        video_output_path
    ]
    try:
        result = subprocess.run(ffmpeg_command, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg Error:\n{e.stderr}")
        raise

    # Clean up temporary file
    os.remove(txt_path)


def save_as_video_clip(audio_samples:np.ndarray, image:np.ndarray, video_output_path:str, expected_sample_length=48_000 * 10) -> None:
    """
    Save a still image and matching audio as a video file.

    # EXAMPLE:
    >> import numpy as np
    >> audio = np.random.randn(48_000 * 10)*0.005  # 10 seconds of random noise
    >> image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)  # Random RGB image
    >> save_as_video_clip(audio, image, "test_output.mp4")

    :param audio_samples: 1D numpy array of raw audio samples (must match expected_sample_length)
    :param image: Image as a numpy array (e.g. from a PIL image converted with np.array)
    :param video_output_path: .mp4 save path to of the video - (other formats might work but are not tested)
    :param expected_sample_length: Expected number of audio samples (default: 10 sec @ 48kHz)
    :return: None
    """
    # Input checks
    S.assert_types(
        [audio_samples, image, video_output_path],
        [np.ndarray, np.ndarray, str],
        ["audio_samples", "image", "video_path"]
    )
    S.assert_positive_int(expected_sample_length, zero_allowed=False)
    assert audio_samples.shape == (expected_sample_length,), f"Expected `{expected_sample_length=}` but found `{audio_samples.shape=}`"
    if not video_output_path.lower().endswith(".mp4"):
        warnings.warn("I have only tested the code with .mp4 files. It may well work with e.g. AVI-files, but perhaps not..")

    # Parameters
    frame_rate = 24
    sample_rate = 48000
    duration = len(audio_samples) / sample_rate  # Duration in seconds

    # Create an audio clip from the numpy array
    def make_audio(t):
        t = np.array(t, ndmin=1)
        idx = (t * sample_rate).astype(int)  # Convert time to sample indices
        idx = np.clip(idx, 0, len(audio_samples) - 1)  # Ensure indices are within bounds
        return audio_samples[idx]
    audio_clip = AudioClip(make_audio, duration=duration, fps=sample_rate)

    # Create video with image and sound
    image_clip = ImageClip(image, duration=duration)
    video_clip = image_clip.set_audio(audio_clip)
    video_clip.write_videofile(video_output_path, fps=frame_rate, logger=None)


#####################################################
# Random
#####################################################

def get_path_valid_timestamp() -> str:
    """
    Returns a timestamp string safe for use in file paths.
    Format: YYYY-MM-DD-HH-MM-SS-microseconds

    EXAMPLE:
        >> get_path_valid_timestamp()
        2025-05-15-10-47-12-413970
    """
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")


class MemoryLogger:
    """
    Simple utility for logging CPU and GPU memory usage to file.

    # EXAMPLE:
    >> logger = MemoryLogger("logs/memory.txt")
    >> logger({"step": 5, "note": "after inference"})

    :param log_path: Path to the file where memory usage will be appended
    """
    def __init__(self, log_path:str):
        S.assert_path_dont_exists(log_path, "log_path")
        self.log_path = log_path

    def __call__(self, extra_info:dict=None) -> None:
        """
        Log current CPU and GPU memory usage to file.

        :param extra_info: Optional dictionary of extra info to include in the log line
        :return: None
        """
        S.assert_type(extra_info, dict, "extra_info", allow_none=True)
        gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # (1024 ** 3) --> Convert to MB
        gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        cpu_memory_used = psutil.virtual_memory().used / (1024 ** 3)
        extra_info_str = ", ".join([f"{k}:{v}" for k, v in extra_info.items()]) if extra_info else ""
        log_data = f"{extra_info_str}, RAMa: {cpu_memory_used:.2f}, VRAMa: {gpu_memory_allocated:.2f}, VRAMr: {gpu_memory_reserved:.2f}"
        with open(self.log_path, "a") as file:
            file.write(log_data + "\n")

def seed_everything(seed:int) -> None:
    """
    Set all relevant random seeds for reproducibility.

    Affects Python, NumPy, PyTorch (CPU and GPU), and disables hash randomization.

    :param seed: Integer seed value
    :return: None
    """
    S.assert_type(seed, int, "seed")
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def is_ffmpeg_available() -> bool:
    """
    Check if `ffmpeg` is available on the system PATH.

    # EXAMPLE:
    >> is_ffmpeg_available()  # True or False

    :return: True if ffmpeg is available, otherwise False
    """
    return shutil.which("ffmpeg") is not None

def assert_ffmpeg_available():
    """
    Raise an error if `ffmpeg` is not available on the system PATH.

    # EXAMPLE:
    >> assert_ffmpeg_available()  # raises if not found

    :return: None
    """
    if not is_ffmpeg_available():
        raise EnvironmentError("FFmpeg is not available on the system PATH.")
