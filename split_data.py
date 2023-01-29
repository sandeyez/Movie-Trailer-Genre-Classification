# NOTE: The code in this file comes from a Google Colab notebook.
# Hence, the structure might be a bit weird, as not everything is
# executed at the same time. 

# This file contains code that is used to split the data into:
# 1. Training data (80%)
# 2. Validation data (10%)
# 3. Test data (10%)

import os
import numpy as np
import pandas as pd
import shutil
import random
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive/',force_remount = True)
PATH = "/content/drive/MyDrive/trailer_classification"
DATA_DIR = os.path.join(PATH, "data/inputs_new")
SPLIT_DIR = os.path.join(PATH, "data/splits")

# Create splits directory if it doesn't exist
os.chdir("/content/drive/MyDrive/trailer_classification/data")
if not os.path.exists('train_new'):
  os.mkdir('train_new')
if not os.path.exists('test_new'):
  os.mkdir('test_new')
if not os.path.exists('val_new'):
  os.mkdir('val_new')

# Split the data into train, test and validation
allFileNames = os.listdir(DATA_DIR)

np.random.seed(123)
np.random.shuffle(allFileNames)

train_FileNames,val_FileNames,test_FileNames = np.split(np.array(allFileNames),[int(len(allFileNames)*0.8),int(len(allFileNames)*0.9)])

# Converting file names from array to list
train_FileNames = [DATA_DIR+'/'+ name for name in train_FileNames]
val_FileNames = [DATA_DIR+'/' + name for name in val_FileNames]
test_FileNames = [DATA_DIR+'/' + name for name in test_FileNames]

print('Total images  : '+ DATA_DIR + ' ' +str(len(allFileNames)))
print('Training : '+ DATA_DIR + ' '+str(len(train_FileNames)))
print('Validation : '+ DATA_DIR + ' ' +str(len(val_FileNames)))
print('Testing : '+ DATA_DIR + ' '+str(len(test_FileNames)))

# Create a function that converts the file names to indices
from pathlib import Path

def filenames_to_indices(fnames):
  return [int(os.path.splitext(Path(file).stem)[0]) for file in train_FileNames]

# This function is used to check through the data and see if there are any corrupted files
from tqdm import tqdm

folders = ["inputs_new"]
corrupted = []
for folder in folders:
  files = os.listdir(folder)
  for file in tqdm(files):
    try:
      print(file)
      np.load(f"{folder}/{file}", allow_pickle=True)
    except OSError as e:
      corrupted.append((folder, file))
      print(e)
print(len(corrupted))

# Get the label distributions for a given list of file names.
import matplotlib.pyplot as plt

def get_label_dist(filenames, label_data=label_data):
  indices = filenames_to_indices(filenames)
  labels = label_data.loc[indices, "action":"thriller"]
  frequencies = labels.sum()
  total = frequencies.sum()
  rel_freq = (frequencies / total * 100).sort_values(ascending=False)
  fig, ax = plt.subplots()
  rel_freq.plot(ax=ax, kind="bar")
  plt.show()

get_label_dist(allFileNames)
get_label_dist(train_FileNames)
get_label_dist(val_FileNames)
get_label_dist(test_FileNames)