import os
import re
import librosa
import string
import math

from typing import List

import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET

import soundfile as sf
from pydub import AudioSegment
from pytube import YouTube

import pandas as pd

from textgrids import TextGrid

def check_existing(vid_fname: str, dataset_dir: str) -> bool:
    """Returns whether a video has been downloaded or not."""
    return os.path.exists(Path(dataset_dir) / vid_fname)

def extract_video_id(url: str) -> str:
    """Extracts the youtube video id from a url with the
    `www.youtube.com/watch?v={id}` format"""
    pattern = r'(?<=v=)[\w-]+'
    match   = re.search(pattern, url)

    if match:
        video_id = match.group(0)
        return video_id
    else:
        return None

def youtube_download(url: str, vid_dir: Path) -> str:
    """Downloads a Youtube video in .mp4 format in the highest
    available resolution."""
    yt = YouTube(url)
    yt.streams\
        .filter(progressive=True, file_extension='mp4')\
        .order_by('resolution')\
        .desc()\
        .first()\
        .download(vid_dir)
    captions = yt.captions
    return captions

def get_audio_feats(audio:          np.ndarray,
                    n_mel_channels: int=128,
                    filter_length:  int=512,
                    win_length:     int=432,
                    hop_length:     int=160,
                    sr:             int=16_000) -> np.ndarray:
    """Return mel-spectrogram feature for audio"""
    audio_features = librosa.feature.melspectrogram(
        audio,
        sr=sr,
        n_mels=n_mel_channels,
        center=False,
        n_fft=filter_length,
        win_length=win_length,
        hop_length=hop_length).T
    audio_features = np.log(audio_features + 1e-5)
    return audio_features.astype(np.float32)

def load_audio(fname: str, mono: bool=True, sr=16_000) -> np.ndarray:
    """Loads and formats audio file into an `np.ndarray`"""
    mp3_file = AudioSegment.from_file(fname, format='mp3')
    wav_file = mp3_file.export(format='wav')
    audio, r = sf.read(wav_file)

    # For mono audio, take first channel only
    if mono and len(audio.shape) > 1:
        audio = audio[:, 0]

    if r != sr:
        audio = librosa.resample(audio, orig_sr = r, target_sr = sr)

    return audio

def load_phonemes(
        textgrid_fname: str,
        audio_feats: np.ndarray,
        phoneme_dict: List[str],
        temporal_scale: int = 100) -> np.ndarray:
    """Returns the pre-processed phonemes for an audio file, for
    the same `seq_len` as the audio features."""
    tg = TextGrid(textgrid_fname)
    phone_intervals = tg["phones"]

    seq_len   = audio_feats.shape[0]
    phone_ids = np.zeros(seq_len, dtype=np.int64)
    phone_ids[phone_ids == 0] = -1

    for interval in phone_intervals:
        xmin = interval.xmin
        xmax = interval.xmax

        phone = interval.text.lower()
        if phone in ["", "sp", "spn"]:
            phone = "sil"
        if phone[-1] in string.digits:
            phone = phone[:-1]
        ph_id = phoneme_dict.index(phone)

        phone_win_start    = int(xmin * temporal_scale)
        phone_duration     = xmax - xmin
        # print("PHONE LOWER THAN INPUT FRAME SIZE:", phone_win_duration < 0.04)

        phone_win_duration = int(math.ceil(phone_duration * temporal_scale))
        phone_win_end      = phone_win_start + phone_win_duration

        phone_ids[phone_win_start:phone_win_end] = ph_id
    
    # ii = np.where(phone_ids == -1)[0]

    assert (phone_ids >= 0).all(), 'missing aligned phones'
    return phone_ids

def load_phoneme_dict(phoneme_dict_path: str) -> List[str]:
    """Convert `phoneme_dict` file into phoneme dictionary list."""
    with open(phoneme_dict_path) as f:
        content = [l.split(":")[1].strip() for l in f.read().split("\n")]
    return content

file_entry = lambda xmin, xmax, items: f"""File type = "ooTextFile"
Object class = "TextGrid"
xmin = {xmin:.3f}
xmax = {xmax:.3f}
tiers? <exists>
size=1
item []:
{items}"""

item_entry = lambda xmin, xmax, intervals, intervals_len: f"""\titem [1]:
\t\tclass = "IntervalTier"
\t\tname = "words"
\t\txmin = {xmin:.3f}
\t\txmax = {xmax:.3f}
\t\tintervals: size = {intervals_len}
{intervals}"""

interval_entry = lambda interval, xmin, xmax, text: f"""\t\t\tintervals [{interval}]:
\t\t\t\txmin = {xmin:.3f}
\t\t\t\txmax = {xmax:.3f}
\t\t\t\ttext = "{text}\""""

def gen_textgrid(metadata: pd.DataFrame) -> str:
    """Create TextGrid from pd.DataFrame"""
    current_tokens = metadata
    texts          = current_tokens["word"]
    xmins          = current_tokens["start"]
    xmaxs          = current_tokens["end"]
    intervals = \
        [interval_entry(i+2, xmin, xmax, text)
        for i, (xmin, xmax, text) in enumerate(zip(xmins, xmaxs, texts))]
    intervals = [interval_entry(1, 0, xmins.iloc[0], "")] + intervals
    intervals_txt = "\n".join(intervals)
    items = item_entry(
        xmin=0,
        xmax=xmaxs.iloc[-1],
        intervals=intervals_txt,
        intervals_len=len(intervals))
    file_content = file_entry(0, xmaxs.iloc[-1], items)

    return file_content