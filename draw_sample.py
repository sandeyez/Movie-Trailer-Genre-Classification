# NOTE: The code in this file comes from a Google Colab notebook.
# Hence, the structure might be a bit weird, as not everything is
# executed at the same time.

# This file is used to draw samples from the full dataset of sizes:
# - 250
# - 500
# - 1000
# - 1500
# - 2000

# import required packages
import pandas as pd
import numpy as np

from google.colab import drive
import os

# mount Google Drive
drive.mount('/content/drive/',force_remount = True)
path = "/content/drive/MyDrive/trailer_classification"
data_dir = os.path.join(path, "data")
os.chdir(data_dir)

# Find all the files for which both audio and video are available
available_files = pd.read_csv('trailer_dl_info.csv')
available_files = available_files.dropna(subset=['audio_downloaded', 'video_downloaded'])
available_files['both_available'] = available_files[['audio_downloaded','video_downloaded']].all(1)
available_files = available_files[available_files.both_available]

both_available = pd.Series(available_files['both_available'])
index = both_available[both_available].index

# Draw the samples and save them to the csv file
sample2000 = both_available.sample(2000, random_state=123)
sample1500 = sample2000.sample(1500, random_state=123)
sample1000 = sample1500.sample(1000, random_state=123)
sample500 = sample1000.sample(500, random_state=123)
sample250 = sample500.sample(250, random_state=123)

available_files.loc[sample2000.index, 'sample2000'] = True
available_files.loc[sample1500.index, 'sample1500'] = True
available_files.loc[sample1000.index, 'sample1000'] = True
available_files.loc[sample500.index, 'sample500'] = True
available_files.loc[sample250.index, 'sample250'] = True
available_files.loc[available_files["sample2000"].isna(), 'sample_all'] = True

available_files.to_csv('trailer_dl_info_update.csv', index=False, header=True, sep=',')
print('success')