# MUSIC MOOD RECOGNITION

A machine learning approach to predict the mood of the music or a song and using that predicted mood to create a customized playlist from a list of available songs in local directory.

## Approach

We have taken 518 songs and extracted 18 features from them using librosa and saved in _dataset.csv_ file. Mean,standard deviation and variance of all those 18 features is calculated as well. By doing that total number of input features in the dataset are 54. To manually label these songs with ease, we have relied on the top spotify playlists which are created according to the moods. Some of them are:

1. Happy Songs

   - [happy hits](https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC)
   - [happy summer beats](https://open.spotify.com/playlist/37i9dQZF1DWSf2RDTDayIx)
   - [wake up happy](https://open.spotify.com/playlist/37i9dQZF1DX0UrRvztWcAU)

2. Relax Playlists

   - [relax & unwind](https://open.spotify.com/playlist/37i9dQZF1DWU0ScTcjJBdj)
   - [deep house relax](https://open.spotify.com/playlist/37i9dQZF1DX2TRYkJECvfC)

3. Angry Playlists

   - [angry songs](https://open.spotify.com/playlist/71Xpaq3Hbpxz6w9yDmIsaH)

4. Sad Playlists
   - [sad songs](https://open.spotify.com/playlist/37i9dQZF1DX7qK8ma5wgG1)
   - [sad beats](https://open.spotify.com/playlist/37i9dQZF1DWVrtsSlLKzro)

The issue of a song belonging to one or more category is resolved by assigning them to a particular mood category out of four.

## Models useds and their accuracies.

| Models | Training accuracy | Testing accuracy |
|------- | ----------------- | ---------------- |
| Logistic Regression | 88.12% | 82.05% |
| Naive Bayes | 78.45% | 76.28% |
| KNN | 78.17% | 78.84% |
| SVM | 87.84% | 76.92% |


## Prerequisites

Python 3.7 or any other version above than Python 2.7

```
https://realpython.com/installing-python/
```

sklearn

```
pip install -U scikit-learn
```

pandas

```
pip install pandas
```

numpy

```
pip install numpy
```

matplotlib

```
pip install matplotlib
```

pickle

```
pip install pickle
```

librosa

```
pip install librosa
```

tkinter

```
apt-get install python-tk
```

pygame

```
pip install pygame
```

mutagen

```
pip install mutagen
```

ttkthemes

```
pip install ttkthemes
```

## How To Run

Just install all the prerequisites and run the musicplayer.py file.
A GUI will open, Add your song and wait for few seconds to get the mood of the song.
After detection of mood, All the songs having similar mood will be added automatically.

## Author

**[Prashant Sharma](https://github.com/shamprashant)**
