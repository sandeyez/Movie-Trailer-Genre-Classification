# NOTE: The code in this file comes from a Google Colab notebook.
# Hence, the structure might be a bit weird, as not everything is
# executed at the same time.

# This file contains code to setup the CNN model and train it.

# import required packages
import torch.nn.functional as F
from torch.nn import init
import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
import copy
import matplotlib.pyplot as plt
import time
import math
import librosa
from librosa import display
from tqdm import tqdm

from random import random
import random as rand
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from google.colab import drive

from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    hamming_loss, 
    precision_score, 
    recall_score,
    f1_score)

# mount Google Drive
drive.mount('/content/drive/',force_remount = True)
path = "/content/drive/MyDrive/trailer_classification"
DATA_DIR = os.path.join(path, "data")
RESULTS_DIR = os.path.join(path, "results")
os.chdir(DATA_DIR)

# Set a random seed for reproducibility
def seed_everything(seed):
    rand.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(123)

# ----------------------------
# Sound Dataset
# ----------------------------
class SoundDS(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.data_path = str(data_path)
        self.duration = 13000
        self.sr = 22050
        self.channel = 1


    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return len(os.listdir(self.data_path))

        # ----------------------------

    # Get i'th item in dataset
    # by first getting the i'th item in the directory
    # getting its name for the index of the class labels
    #
    # ----------------------------
    def __getitem__(self, file_idx):
        # Absolute file path of the mfcc file - concatenate the audio directory with
        # the relative path

        # the data at the i'th item in the directory
        data_at_file_idx = os.listdir(self.data_path)[file_idx]

        # the data name is the index in the df
        data_idx = int(Path(data_at_file_idx).stem)

        #path to mfcc file
        mfcc_file = os.sep.join([self.data_path, data_at_file_idx])
        mfcc = np.load(mfcc_file)

        # Get the Class ID array
        class_id = self.df.loc[data_idx-1].to_numpy().astype(float)

        return mfcc, class_id
    
# Scale data
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([224,224]),
    transforms.Normalize(mean=-3, std=39)
])

def get_datapoint(df_index, df, data_path="inputs_new", transform=None):
    # Absolute file path of the mfcc file - concatenate the audio directory with
    # the relative path

    # the data at the i'th item in the directory
    filename = f"{df_index + 1}.npy"

    #path to mfcc file
    mfcc_path = os.path.join(data_path, filename)

    mfcc = np.load(mfcc_path)

    # Get the Class ID array
    class_id = df.loc[df_index].to_numpy().astype(float)

    # Transform if necessary
    if transform:
        return transform(mfcc), class_id # return data, label (X, y)
    else:
        return mfcc, class_id # return data, label (X, y)

# Model training
# ----------------------------
# Hyperparameters
# ----------------------------
min_epochs = 10
batch_size = 32 # 32 for our model

# ----------------------------
# Dataloaders
# ----------------------------
RANDOM_STATE = 123
SAMPLE_SIZE = 4750

TRAIN_DIR = os.path.join(DATA_DIR, "train_new")
VAL_DIR = os.path.join(DATA_DIR, "val_new")
TEST_DIR = os.path.join(DATA_DIR, "test_new")

df = pd.read_csv("mtgc.csv", sep=',')
df = df.drop(columns=['mid', 'split0', 'split1', 'split2'])
downloaded_from_df = [int(os.path.splitext(file)[0]) - 1 for file in  os.listdir("inputs_new")]
df = df.loc[downloaded_from_df]
print(len(downloaded_from_df))
sampled_indices = df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE).index.values

train, val, test = np.split(np.array(sampled_indices),[int(len(sampled_indices)*0.8),int(len(sampled_indices)*0.9)])

ds_train = [get_datapoint(file_idx, df, transform=data_transforms) for file_idx in tqdm(train)]
ds_val = [get_datapoint(file_idx, df, transform=data_transforms) for file_idx in tqdm(val)]
ds_test = [get_datapoint(file_idx, df, transform=data_transforms) for file_idx in tqdm(test)]

datasets = {}
datasets['train'] = ds_train
datasets['val'] = ds_val
datasets['test'] = ds_test

dataloaders = {x: torch.utils.data.DataLoader(datasets[x], 
                                              batch_size=batch_size, 
                                              num_workers=8, 
                                              pin_memory=True,
                                              shuffle=True) for x in ['train','val','test']}
                                             
# ----------------------------
# Audio Classification Model
# ----------------------------
class AudioClassifier(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 5), stride=2, padding=2)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Second Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Second Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=10)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)

    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        # Final output
        return x


# Create the model and put it on the GPU if available
myModel = AudioClassifier()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)
# Check that it is on Cuda
next(myModel.parameters()).device

# Compute the Hamming score
def hamming_score(y_true, y_pred):

    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
        
    return np.mean(acc_list)

# ----------------------------
# Training Loop
# ----------------------------
def training(model, dataloaders, min_epochs, learn_rate, wgt_decay, early_stop: bool=True, save_to="some_string.pth"):

    since = time.time()

    # create storage
    train_loss_history = []
    train_acc_history = []

    val_loss_history = []
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    counter = 0
    best_loss = 10000.0
    best_acc = 0
    emr = 0
    hamm_loss = 0
    recall = 0
    precision = 0
    f1score = 0
    frequencies = np.array([10.37, 6.5, 13.97, 8.27, 19.87, 5.22, 8.57, 7.30, 5.11, 14.82]) / 100
    pos_weights = (1 - frequencies) / frequencies

    #  weights for : ["action", "adventure", "comedy", "crime", "drama", "fantasy", "horror", "romance", "sci-fi", "thriller"]
    pos_weight = torch.tensor(pos_weights).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=wgt_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learn_rate * 5,
                                                    steps_per_epoch=int(len(dataloaders['train'].dataset)),
                                                    epochs=50,
                                                    anneal_strategy='linear')

    for epoch in range(int(math.pow(min_epochs, 10))):
        print('Epoch {}'.format(epoch))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0
            correct_preds = 0
            total_loading_time = 0

            # track time to perform data loading/model training
            start_epoch = time.time()
            start_loading = time.time()

            # Repeat for each batch in the training set     
            for batches, data in enumerate(dataloaders[phase]):

                loading_time = (time.time() - start_loading)
                total_loading_time = total_loading_time + loading_time

                # Get the input features and target labels, and put them on the GPU
                inputs, labels = data[0].to(device), data[1].to(device)

                # Normalize the inputs
                #inputs_m, inputs_s = inputs.mean(), inputs.std()
                #inputs = (inputs - inputs_m) / inputs_s

                # Zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    outputs = torch.sigmoid(outputs).cpu()
                    preds = np.round(outputs.detach())

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                # Keep stats for Loss and Accuracy
                running_loss += loss.item()

                # calculate performance metrics
                epoch_acc = hamming_score(labels.cpu(), preds)

                # reset loading time
                start_loading = time.time()

            training_time = (time.time() - start_epoch - total_loading_time)
            print('Data loading took {:.0f}m {:.0f}s'.format(total_loading_time // 60, total_loading_time % 60))
            print('Model training/evaluation took {:.0f}m {:.0f}s'.format(training_time // 60, training_time % 60))

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            #epoch_acc = correct_preds.double() / total_preds

            print('{} Loss: {:.4f} Hamming Score: {:.4f}'.format(phase, epoch_loss, epoch_acc.item()))

            # safe training loss and accuracy
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)

            # retain loss and acc of best model
            if phase == 'val' and epoch_loss < best_loss:
                # store performance metrics
                best_loss = epoch_loss
                best_acc = epoch_acc
                emr = accuracy_score(labels.cpu(), preds, normalize=True, sample_weight=None)
                hamm_loss = hamming_loss(labels.cpu(), preds)
                precision = precision_score(labels.cpu(), preds, average='samples')
                recall = recall_score(labels.cpu(), preds, average='samples')
                f1score = f1_score(labels.cpu(), preds, average='samples')
                
                # reset counter
                counter = 0
                # deep copy the model
                best_model_wts = copy.deepcopy(model.state_dict())

            elif phase == 'val' and epoch_loss >= best_loss:
                counter += 1

            # safe validation loss and accuracy
            if phase == 'val':
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc) 

        # introduce early stopping
        if early_stop == True:
            if epoch > min_epochs - 1 and counter >= 10:
                break
        
        # load best model weights
        model.load_state_dict(best_model_wts)
        # save and print best model
        torch.save(model.state_dict(best_model_wts), os.path.join(RESULTS_DIR, save_to))


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best model loss: {:4f}, Hamming Score: {:4f}, Exact Match Ratio: {:4f}, Hamming Distance: {:4f}, Precision: {:4f}, Recall: {:4f}, F1-Score: {:4f}'.format(best_loss, best_acc, emr, hamm_loss, precision, recall, f1score))

    return model, train_loss_history, train_acc_history, val_loss_history, val_acc_history

# Create the model and put it on the GPU if available
myModel = AudioClassifier()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)
# Check that it is on Cuda
next(myModel.parameters()).device

learn_rate = 0.0001
wgt_decay = 0.0001

myModelTrained, train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = training(myModel, dataloaders, min_epochs, learn_rate, wgt_decay, early_stop=True, save_to="best_model9000.pth")

from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
BEST_MODEL = "best_model9000"

def inference(model, test_dl, modelfile="best_model9000.pth", conf_prefix = "something"):
    correct_prediction = 0
    total_preds = 0

    best_weights = torch.load(os.path.join(RESULTS_DIR, modelfile))
    model.load_state_dict(best_weights)

    model.eval()

    all_preds = []
    all_labels = []

    # Disable gradient updates
    with torch.no_grad():
        for data in test_dl:
            inputs, labels = data[0].to(device), data[1].to(device)
            all_labels.append(labels.cpu())
            # run the model on the test set to predict labels
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
            preds = np.round(outputs.cpu())
            all_preds.append(preds)

    true = torch.cat(all_labels, dim=0)
    y_pred = torch.cat(all_preds)
    acc = hamming_score(true, y_pred)
    emr = accuracy_score(true, y_pred, normalize=True, sample_weight=None)
    hamm_loss = hamming_loss(true, y_pred)
    precision = precision_score(true, y_pred, average='samples')
    recall = recall_score(true, y_pred, average='samples')
    f1score = f1_score(true, y_pred, average='samples')
    print('Hamming Score: {:4f}, Exact Match Ratio: {:4f}, Hamming Distance: {:4f}, Precision: {:4f}, Recall: {:4f}, F1-Score: {:4f}'.format(acc, emr, hamm_loss, precision, recall, f1score))
    return true, y_pred

true, pred = inference(myModel, dataloaders['test'])

labels = ["action", "adventure", "comedy", "crime", "drama", "fantasy", "horror", "romance", "sci-fi", "thriller"]
for matrix, genre in zip(multilabel_confusion_matrix(true, pred), labels):
  disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
  fig, ax = plt.subplots(figsize=(10, 10))
  plt.rcParams.update({'font.size': 16})
  ax.set_title(f"{genre}")
  plt.title = genre
  disp.plot(cmap="gray", ax=ax)
  plt.savefig(os.path.join(RESULTS_DIR, f"{genre}_best_model9000.png"), dpi=300)
  plt.show()

  # Plot the loss over number of training epochs 

def plot_loss(train_loss_hist, val_loss_hist):
  
  train_loss = []
  train_loss = [epoch for epoch in train_loss_hist]
  
  val_loss = []
  val_loss = [epoch for epoch in val_loss_hist]
  plt.figure(figsize=(10, 10))
  plt.rcParams['font.size']=16
  plt.xlabel("Training Epochs", fontsize=20, labelpad=10)
  plt.ylabel("Loss", fontsize=20, labelpad=10)
  plt.plot(range(0, len(train_loss_hist)), train_loss, color="dimgray", label="Training", linewidth=2)
  plt.plot(range(0, len(train_loss_hist)), val_loss, color="darkgray", label="Validation", linewidth=2)
  # plt.ylim((0,0.05))
  plt.xticks(np.arange(0, len(train_loss_hist), 5))
  plt.legend(loc="top right")
  
  plt.savefig(os.path.join(RESULTS_DIR, 'loss_hist_best_resnet.png'), dpi=300, bbox_inches="tight")

plot_loss(train_loss, val_loss)

# Plot the hamming score over number of training epochs 
def plot_acc(train_acc_hist, val_acc_hist):
  
  train_acc = []
  train_acc = [epoch for epoch in train_acc_hist]
  
  val_acc = []
  val_acc = [epoch for epoch in val_acc_hist]
  plt.figure(figsize=(10, 10))
  plt.rcParams['font.size']=16
  plt.xlabel("Training Epochs", fontsize=20, labelpad=10)
  plt.ylabel("Hamming Score (Accuracy)", fontsize=20, labelpad=10)
  
  plt.plot(range(0, len(train_acc_hist)), train_acc, color="dimgray", label="Training", linewidth=2)
  plt.plot(range(0, len(train_acc_hist)), val_acc, color="darkgray", label="Validation", linewidth=2)
  #plt.ylim((0,1))
  plt.xticks(np.arange(0, len(train_acc_hist), 5))
  plt.legend(loc="top right")
  
  plt.savefig(os.path.join(RESULTS_DIR, 'acc_hist_best_resnet.png'), dpi=300, bbox_inches="tight")

plot_acc(train_acc, val_acc)

from torchvision.models import resnet152, ResNet152_Weights

# myModel.cpu()

num_input_channel = 1

resnet = resnet152(weights=ResNet152_Weights.DEFAULT)
resnet.conv1 = nn.Conv2d(num_input_channel, 64, kernel_size=7, stride=2, padding=3,bias=False)
resnet.fc = nn.Linear(2048, 10)

# Create the model and put it on the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = resnet.to(device)

learn_rate = 0.00001
wgt_decay = 0.001

resnetTrained, train_loss, train_acc, val_loss, val_acc = training(resnet, dataloaders, min_epochs, learn_rate, wgt_decay, early_stop=True, save_to="best_resnet9000.pth")

true, pred = inference(resnetTrained, dataloaders['test'], modelfile="best_resnet9000.pth")

labels = ["action", "adventure", "comedy", "crime", "drama", "fantasy", "horror", "romance", "sci-fi", "thriller"]
for matrix, genre in zip(multilabel_confusion_matrix(true, pred), labels):
  disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
  fig, ax = plt.subplots(figsize=(10, 10))
  plt.rcParams.update({'font.size': 16})
  ax.set_title(f"{genre}")
  plt.title = genre
  disp.plot(cmap="gray", ax=ax)
  plt.savefig(os.path.join(RESULTS_DIR, f"{genre}_best_resnet9000.png"), dpi=300)
  plt.show()