import os

import matplotlib.pyplot as plt

import librosa


def func(cls):
    img_names = os.listdir("genres/" + cls)
    os.makedirs("wavelets/train/" + cls)
    os.makedirs("wavelets/test/" + cls)
    print(cls)
    train_names = img_names[:60]
    test_names = img_names[60:]
    cnt = 0
    for nm in train_names:
        cnt += 1
        x, sr = librosa.load("genres/" + cls + "/" + nm)
        # plt.figure(figsize=(14, 5))
        librosa.display.waveshow(x)
        plt.savefig("wavelets/train/" + cls + "/" + str(cnt) + ".png")
        plt.close()

    cnt = 0
    for nm in test_names:
        cnt += 1
        x, sr = librosa.load("genres/" + cls + "/" + nm)
        # plt.figure(figsize=(14, 5))
        librosa.display.waveshow(x)
        plt.savefig("wavelets/test/" + cls + "/" + str(cnt) + ".png")
        plt.close()


classes = [
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
for c_name in classes:
    func(c_name)
