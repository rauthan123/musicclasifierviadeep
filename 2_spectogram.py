import os

import matplotlib.pyplot as plt

import librosa
import librosa.display


def func1(cls):
    img_names = os.listdir("genres/" + cls)
    os.makedirs("spectrogram/train/" + cls)
    os.makedirs("spectrogram/test/" + cls)
    print(cls)
    train_names = img_names[:60]
    test_names = img_names[60:]
    cnt = 0
    for nm in train_names:
        cnt += 1
        x, sr = librosa.load("genres/" + cls + "/" + nm)
        X = librosa.stft(x)
        Xdb = librosa.amplitude_to_db(abs(X))
        librosa.display.specshow(Xdb)
        plt.savefig("spectrogram/train/" + cls + "/" + str(cnt) + ".png")
        plt.close()

    cnt = 0
    for nm in test_names:
        cnt += 1
        x, sr = librosa.load("genres/" + cls + "/" + nm)
        X = librosa.stft(x)
        Xdb = librosa.amplitude_to_db(abs(X))
        librosa.display.specshow(Xdb)
        plt.savefig("spectrogram/test/" + cls + "/" + str(cnt) + ".png")
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
    func1(c_name)
