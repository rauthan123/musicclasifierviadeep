import os
import pickle

from cv2 import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
img_size = 256


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

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    # rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.2,  # Randomly zoom image
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    # horizontal_flip = True,  # randomly flip images
    vertical_flip=False,
)  # randomly flip images

datagen.fit(x_train)

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

history = model.fit(x_train, y_train, epochs=500, validation_data=(x_val, y_val))

model.save_weights("model_train/500_epoch_simple_lr.cpkt")

pickle.dump(history.history, open("model_train/history_500_epoch_simple.pkl", "wb"))

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(500)

plt.figure(figsize=(25, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.show()

history = pickle.load(open("model_train/history_500_epoch_simple.pkl", "rb"))
acc = history["accuracy"]
val_acc = history["val_accuracy"]
loss = history["loss"]
val_loss = history["val_loss"]

epochs_range = range(500)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
plt.rc("xtick", labelsize=10)  # fontsize of the tick labels
plt.rc("ytick", labelsize=10)
ax1.plot(epochs_range, acc, label="Training Accuracy", c="#4CAF50", linewidth=4)
ax1.plot(epochs_range, val_acc, label="Validation Accuracy", c="red", linewidth=4)
ax1.legend()
ax1.set_title("Training and Validation Accuracy", fontsize=18)
ax1.set_ylabel("Accuracy", fontsize=18)
ax1.set_xlabel("Epoch", fontsize=18)

ax2.plot(epochs_range, loss, label="Training Loss", c="#4CAF50", linewidth=4)
ax2.plot(epochs_range, val_loss, label="Validation Loss", c="red", linewidth=4)
ax2.legend()
ax2.set_title("Training and Validation Loss", fontsize=18)
ax2.set_ylabel("Loss", fontsize=18)
ax2.set_xlabel("Epoch", fontsize=18)
fig.tight_layout(pad=3.0)
# plt.show()
plt.savefig("sim_plot1.png", bbox_inches="tight")
plt.clf()
