# Toy-Shazam

![68747470733a2f2f7777772e636f646570726f6a6563742e636f6d2f4b422f5750462f6475706c6963617465732f636f6e6365707469616c6f766572766965772e706e67-1](https://user-images.githubusercontent.com/91341004/151998516-2c311377-a9a6-4d09-8339-6eba83cc8b59.png)

We will implement a simplified version of [Shazam](https://www.shazam.com) by dealing with hashing algorithms. In particular, we will implement a LSH algorithm that takes as input an audio track and finds relevant matches. Then we will play with a dataset gathering songs from the International Society for Music Information Retrieval Conference. The tracks (songs) include much information. We will focus on the track information, features (extracted with librosa library from Python) and audio variables provided by Echonest (now Spotify).

### What's in here?
There are two branches: `main` and `files`. In the branch `files` there are
- the datasets needed for the second part of the project `echonest.csv`, `features.csv` and `tracks.csv`
- the dataset needed for the first part of the project `queries` 
In the branch `main` there are
- `kmeans_utils.py` that contains utility functions to run an optimized version of the k-means algorithm
- the notebook with all the code of the project `main.ipynb`
- the model we used saved as pickle file `model.pkl` 


### Dataset
We will use the following datasets:
- *queries* for the first part of the notebook (available in the *files* branch of this repository)
- *mp3s32k* for the first part of the notebook (available at https://www.kaggle.com/dhrumil140396/mp3s32k)
- *echonest.csv*, *features.csv* and *tracks.csv* for the second part of the notebook (available in the *files* branch of this repository)


### Disclamair
This project was part of a course of Algorithmic Methods of Data Mining at Sapienza University of Rome held by Prof. Aris Anagnostopoulos. This project was done in collaboration with Matteo Broglio and Petar Ulev.


