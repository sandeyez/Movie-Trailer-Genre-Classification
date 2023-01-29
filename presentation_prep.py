# NOTE: The code in this file comes from a Google Colab notebook.
# Hence, the structure might be a bit weird, as not everything is
# executed at the same time.

# This file contains code used to prepare the data for the presentation.

import os

from google.colab import drive

drive.mount('/content/drive/',force_remount = True)
path = "/content/drive/MyDrive/trailer_classification"
os.chdir(path)

# Get the lengths of the total, train, test and validation sets, respectively.
total = len(os.listdir("data/inputs"))
train = len(os.listdir("data/train"))
test = len(os.listdir("data/test"))
validation = len(os.listdir("data/val"))

print(total, train, test, validation)
print(train / total * 100)
print(test / total * 100)
print(validation / total * 100)