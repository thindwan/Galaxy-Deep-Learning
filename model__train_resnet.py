import cv2
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from skimage.transform import resize
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
import cv2
import warnings
warnings.filterwarnings("ignore")
import os
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from torch.autograd import Variable
import warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F
import torch.utils.data as utils
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, BCELoss, BCEWithLogitsLoss
from torch.optim import Adam, SGD
from sklearn.metrics import accuracy_score
from torchvision import transforms, datasets, models

# %% ===================================================================================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# print(device)

# % =========================================HYPER PARAMETERS============================================================
RESIZE_TO = 64
DROPOUT = 0.2
N_EPOCHS = 150
LR = 0.000001
# %% ===================================================================================================================

df = pd.read_csv('/home/ubuntu/.kaggle/training_solutions_rev1.csv')
y = pd.DataFrame(df)
y = y.to_numpy()


ORIG_SHAPE = (424, 424)
CROP_SIZE = (256, 256)
IMG_SHAPE = (28, 28)

from sklearn.metrics import mean_squared_error

def get_rmse(true_labels, pred_labels):
    # print(np.shape(true_labels),type(true_labels))
    # print(np.shape(pred_labels), type(pred_labels))
    rmse = np.sqrt(mean_squared_error(true_labels, pred_labels))
    # print('Train RMSE',rmse)
    return rmse

def data(dataframe):
    x1 = (424 - 256) // 2
    y1 = (424 - 256) // 2
    sel = dataframe.values
    ids = sel[:, 0].astype(int).astype(str)
    y_batch = sel[:, 1:]
    x_batch = []
    for i in ids:
        image = cv2.imread('/home/ubuntu/.kaggle/images_training_rev1/' + i + '.jpg')

        image = image[x1:x1 + 256, y1:y1 + 256]
        image = cv2.resize(image, IMG_SHAPE) / 255
        x_batch.append(image)
    x_batch = np.array(x_batch)
    return x_batch



x = data(df)
# print(x.shape)
# print(x[1].shape)

# one-hot encode the multi labels using MultiLabelBinarizer()
mlb = MultiLabelBinarizer()
labels = [["Class1.1", "Class1.2", "Class1.3", "Class2.1",
               "Class2.2", "Class3.1", "Class3.2", "Class4.1", "Class4.2",
               "Class5.1", "Class5.2", "Class5.3", "Class5.4", "Class6.1",
               "Class6.2", "Class7.1", "Class7.2", "Class7.3", "Class8.1",
               "Class8.2", "Class8.3", "Class8.4", "Class8.5", "Class8.6",
               "Class8.7", "Class9.1", "Class9.2", "Class9.3", "Class10.1",
               "Class10.2", "Class10.3", "Class11.1", "Class11.2", "Class11.3",
               "Class11.4","Class11.5", "Class11.6"]]

mlb.fit(labels)
y = mlb.transform(y)
# print(y.shape)
# print(y[1:5])

# # %% ========================================DATA PREPARE=============================================================


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)


# converting training and testing images into torch format
x_train = torch.from_numpy(x_train).float().permute(0, 3, 1, 2)
x_test = torch.from_numpy(x_test).float().permute(0, 3, 1, 2)

# converting the training target into torch format
y_train = y_train.astype(int);
y_train = torch.from_numpy(y_train).to(device)


# converting the target into torch format
y_test = y_test.astype(int);
y_test = torch.from_numpy(y_test).to(device)


# shape of training data
print("Shape of tensor converted x_train", x_train.shape), print("Shape of tensor converted y_train", y_train.shape)
# shape of test data
print("Shape of tensor converted x_test",x_test.shape), print("Shape of tensor y_test",y_test.shape)
# print(x_train.dtype), print(y_train.dtype)


# %% -------------------------------------------ResNet50---------------------------------------------------------------
#
model=torchvision.models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 37)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# defining the loss function
criterion = nn.MSELoss()

# checking if GPU is available
# if torch.cuda.is_available():
#     model.cuda()
    # criterion = criterion.cuda().float()

print(model)

# =====================================================
print("Starting training loop...")
BATCH_SIZE = 256
training_loss = []
validation_loss = []
true_labels=[]
pred_labels =[]
training_rmse = []
testing_rmse =[]
for epoch in range(N_EPOCHS):
    loss_train = 0
    model.train().to(device)

    for batch in range(len(x_train)//BATCH_SIZE + 1):
        inds = slice(batch*BATCH_SIZE, (batch+1)*BATCH_SIZE)
        optimizer.zero_grad()
        logits = model(x_train[inds].to(device))
        # _, preds = torch.max(logits, 1)
        loss = criterion(torch.sigmoid(logits), y_train[inds].float())
        pred_labels.append(torch.sigmoid(logits).detach().cpu())
        true_labels.append(y_train[inds].detach().cpu())
        # print("preds",logits)
        # print("preds shape",logits.shape)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()

    training_loss.append(loss_train)
    # print("pred labels",pred_labels)
    train_rmse = get_rmse(torch.cat(true_labels).numpy(),torch.cat(pred_labels).numpy())
    training_rmse.append(train_rmse)
    print("Epoch {} --> Train Loss {:.5f}".format(epoch, loss_train))

    print("Training RMSE",train_rmse)
    # print(training_loss)
    torch.cuda.empty_cache()


    with torch.no_grad():
        model.eval()
        test_true_labels = []
        test_pred_labels = []
        y_test_pred = model(x_test.to(device))
        loss = criterion(torch.sigmoid(y_test_pred), y_test.float())

        test_true_labels.append(y_test.detach().cpu())
        test_pred_labels.append(torch.sigmoid(y_test_pred).detach().cpu())

        loss_test = loss.item()
    validation_loss.append(loss_test)
    test_rmse = get_rmse(torch.cat(test_true_labels).numpy(), torch.cat(test_pred_labels).numpy())
    testing_rmse.append(test_rmse)
    print("Test rmse",test_rmse)

    torch.save(model.state_dict(), "model_resnet.pt")
    print("Epoch {} --> Test Loss {:.5f} "+str(loss_test))
    # print(validation_loss)
    torch.cuda.empty_cache()

# +++++++++++++++++++++++++++++ PLOTTING VALIDATION AND TESTING LOSSES FOR NUMBER OF EPOCHS ====++++++++++++++++++++++++

plt.plot(training_loss, label='Training loss')
plt.plot(validation_loss, label='Validation loss')
plt.title("Train/Validation Losses:batch size=256,lr=0.000001,opt =Adam,epochs=150")
plt.xlabel("Number of Epochs")
plt.ylabel("Training and Validation Losses")
plt.legend()
plt.show()

# ++++++++++++++++++++++++++++++++++++++++++++    RMSE  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

plt.plot(training_rmse, label='Training loss')
plt.plot(testing_rmse, label='Validation loss')
plt.title("RMSE:Training/Validation:batch size=256,lr=0.000901,opt=Adam,epochs=150")
plt.xlabel("Number of Epochs")
plt.ylabel("Training and Validation - RMSE Loss")
plt.legend()
plt.show()