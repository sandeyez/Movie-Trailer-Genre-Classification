# NOTE: The code in this file comes from a Google Colab notebook.
# Hence, the structure might be a bit weird, as not everything is
# executed at the same time.

# This file is used to convert audio files from the drive to MFCCs.
import os
from random import random

import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive
import librosa
from librosa import display
import pandas as pd
from tqdm import tqdm

drive.mount('/content/drive/',force_remount = True)
path = "/content/drive/MyDrive/trailer_classification"
os.chdir(path)

def open_mp3(mp3_file, sampling_rate = 22050, to_mono = True):
    (sig, sampling_rate) = librosa.load(mp3_file, sr=sampling_rate, mono=to_mono)
    return sig

# ----------------------------
# Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
# according to https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5
# ----------------------------

def pad_trunc(sig, sampling_rate, max_duration):
    sig_len = len(sig)
    max_len = sampling_rate // 1000 * max_duration

    if sig_len > max_len:
        # Truncate the signal to the given length
        sig = sig[:max_len]

    elif sig_len < max_len:
        sig = librosa.util.pad_center(sig, max_len)

    return sig

# ----------------------------
# Generate a mfcc
# ----------------------------

def convert_to_mfcc(sig, sr, n_mels=64, n_fft=1024, hop_len=None, show_plot=False, save: bool = False):
    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
    mfcc = librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=40, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)
    if show_plot:
        S = librosa.feature.melspectrogram(y=sig, sr=sr, n_mels=128,
                                            fmax=8000)
        fig, ax = plt.subplots(nrows=2, sharex=False)
        img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                                        x_axis='time', y_axis='mel', fmax=8000,
                                        ax=ax[0])
        fig.colorbar(img, ax=[ax[0]])
        ax[0].set(title='Mel spectrogram')
        ax[0].label_outer()
        img = librosa.display.specshow(mfcc, x_axis='time', ax=ax[1])
        fig.colorbar(img, ax=[ax[1]])
        ax[1].set(title='MFCC')

        if save:
          fig.savefig("presentation_mfcc.png", dpi=350)
        if show_plot:
          plt.show()
    return mfcc

import time
from pydub import AudioSegment

i = 2000
DOWNLOAD_DIR = "downloads"
AUDIO_DIR = "audio"
SAMPLING_RATE = 44100
MAX_DURATION = 130000
start = time.time()
file_path = os.path.join(DOWNLOAD_DIR, AUDIO_DIR, f"{i}.mp3")
sig = open_mp3(file_path, sampling_rate = SAMPLING_RATE)
print(sig.shape)

# end = time.time()
trunc_sig = pad_trunc(sig, SAMPLING_RATE, MAX_DURATION)
convert_to_mfcc(trunc_sig, SAMPLING_RATE, show_plot=True)
end=time.time()
print(end-start)

import time
from pydub import AudioSegment
import logging
import sys

DOWNLOAD_DIR = "downloads"
AUDIO_DIR = "audio"
BASE_SAMPLING_RATE = 44100
SAMPLING_RATE = 44100
MAX_DURATION = 121162

def mp3_to_mfcc(index: int, resample: bool =False, verbose: bool = False, show_plot: bool =False, save: bool =False):
  logger = logging.getLogger("mfcc_generator")

  file_path = os.path.join(DOWNLOAD_DIR, AUDIO_DIR, f"{index}.mp3")
  logger.info("Starting conversion of mp3 to mfcc: path %s", file_path)
  start = time.time()
  raw_signal = AudioSegment.from_file(file_path, "mp4", channels=2, sample_width=4)

  channel_sounds = raw_signal.split_to_mono()
  samples = [s.get_array_of_samples() for s in channel_sounds]

  fp_arr = np.array(samples).T.astype(np.float32)
  fp_arr /= np.iinfo(samples[0].typecode).max
  sig = np.array(fp_arr).astype(np.float32)
  sig = sig.mean(axis=1)
  if resample:
    sig = librosa.resample(sig.T, BASE_SAMPLING_RATE, SAMPLING_RATE, res_type = "zero_order_hold")
  end = time.time()
  trunc_sig = pad_trunc(sig, SAMPLING_RATE, MAX_DURATION)
  mfcc = convert_to_mfcc(trunc_sig, SAMPLING_RATE, show_plot=show_plot, save=save)
  logger.info("Succesfully converted! Time elapsed %s seconds", end - start)
  return mfcc

sample_input_path = "data/trailer_dl_info_update.csv"
SAMPLE_PATH = "data/samples"
if not os.path.exists(SAMPLE_PATH):
  os.mkdir(SAMPLE_PATH)
data = pd.read_csv(sample_input_path).set_index("index")

def save_sample(sample_name, data=data):
  sample = data[~data[sample_name].isna()].index.values
  np.save(os.path.join(SAMPLE_PATH, sample_name), sample)

def load_sample(sample_name):
  logger = logging.getLogger("mfcc_generator")
  sample_path = os.path.join(SAMPLE_PATH, f"{sample_name}.npy")
  logging.info("Loading sample file %s", sample_path)
  sample =  np.load(sample_path)
  logging.info("Succesfully loaded sample file with %s entries", len(sample))
  return sample

INPUT_PATH = "data/inputs"

from tqdm import tqdm
logging.basicConfig(filename="mfcc_transcode.log", level=logging.INFO)
def create_mfcc(mfcc_index, input_path=INPUT_PATH):
    logger = logging.getLogger("mfcc_generator")
    mfcc = mp3_to_mfcc(mfcc_index)
    path_out = os.path.join(input_path, f"{mfcc_index}")
    logger.info("Writing mfcc %s to path %s.npy", mfcc_index, path_out)
    np.save(path_out, mfcc)
    logger.info("Succesfully wrote out mfcc!")


def create_all_mfcc(sample_name):
  logger = logging.getLogger("mfcc_generator")
  logger.info("Starting creating mfcc's for %s", sample_name)
  sample = load_sample(sample_name)
  logger.info("Checking mfcc's that were already created...")
  created_mfccs = set([int(os.path.splitext(mfcc_path)[0]) for mfcc_path in os.listdir(INPUT_PATH)])
  logger.info("Out of the full sample size %s, %s were already created.", len(sample), len(created_mfccs))

  to_create = set(sample) - created_mfccs
  logger.info("Creating %s mfcc's", len(to_create))
  for index in tqdm(to_create):
    create_mfcc(index)
      

create_all_mfcc("sample_all")

import os
sum(os.path.getsize(f) for f in os.listdir('data/inputs_new') if os.path.isfile(f))

sample = load_sample("sample2000")
INPUT_PATH = "data/inputs"

def load_mfcc(mfcc_id):
   return np.load(os.path.join(INPUT_PATH, f"{mfcc_id}.npy"))


def iter_sample(sample_name):
  sample = load_sample(sample_name)
  for i in sample:
    yield load_mfcc(i)

old_folders = ["train", "test", "val"]
new_folders = ["train_new", "test_new", "val_new"]

for old, new in zip(old_folders, new_folders):
  old_files = set(os.listdir("data/" + old))
  new_files= set(os.listdir("data/" + new))
  print(len(new_files - old_files))
  print(len(new_files), len(old_files))

folders = ["train_new"]
corrupted = []
for folder in folders:
  files = os.listdir("data/" + folder)
  for file in tqdm(files):
    try:
      # print(file)
      np.load(f"data/{folder}/{file}", allow_pickle=True)
    except OSError as e:
      corrupted.append((folder, file))
      print(e)
      print(folder, file)
      
from tqdm import tqdm

path = "downloads/audio"
max = 0

for f_path in tqdm(os.listdir(path)):
  filepath = os.path.join(path, f_path)
  audio = AudioSegment.from_file(file_path, "mp4", channels=2, sample_width=4)
  if len(audio) > max:
    max = len(audio)

print(max)

path = "downloads/audio"
max = 0

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
inputs_new = "data/inputs_new"

if not os.path.exists(inputs_new):
  os.mkdir(inputs_new)

sample = [int(os.path.splitext(f_path)[0]) for f_path in os.listdir(path)]

logger = logging.getLogger("mfcc_generator")
logger.setLevel(logging.INFO)
created_mfccs = set([int(os.path.splitext(mfcc_path)[0]) for mfcc_path in os.listdir(inputs_new)])
logger.info("Out of the full sample size %s, %s were already created.", len(sample), len(created_mfccs))

to_create = set(sample) - created_mfccs
logger.info("Creating %s mfcc's", len(to_create))

for audio_id in tqdm(to_create):
  create_mfcc(audio_id, inputs_new)