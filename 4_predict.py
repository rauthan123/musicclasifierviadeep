import os

import numpy as np
import pandas as pd
from cv2 import cv2
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn


img_size = 256

labels = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]


def get_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[
                          ..., ::-1
                          ]  # convert BGR to RGB format
                resized_arr = cv2.resize(
                    img_arr, (img_size, img_size)
                )  # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)


train = get_data("spectrogram/train")
val = get_data("spectrogram/test")

x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train:
    x_train.append(feature)
    y_train.append(label)

for feature, label in val:
    x_val.append(feature)
    y_val.append(label)

# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

model = Sequential()
model.add(Conv2D(32, 3, padding="same", activation="relu", input_shape=(256, 256, 3)))
model.add(MaxPool2D())

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.summary()

opt = Adam(lr=0.0001)
model.compile(
    optimizer=opt,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# trained model
model.load_weights("model_train/500_epoch_simple_lr.cpkt")

predictions = model.predict_classes(x_val)
# predictions = model.predict(x_val)
predictions = predictions.reshape(1, -1)[0]
print(classification_report(y_val, predictions, target_names=labels))

cm1 = confusion_matrix(y_val, predictions)
df_cm = pd.DataFrame(cm1, index=[i for i in labels], columns=[i for i in labels])
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, cmap="RdPu")
plt.savefig("predict/confusion_mrtx1.png", bbox_inches="tight")
