import numpy as np
import simpleaudio as sa
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from configs.config import PARAM_GRID,LOGISTIC_PARAMS,CV_SPLITS
import threading
import time


def beep():
    frequency = 1000  # Hz
    duration = 0.2    # seconds
    fs = 44100        # sample rate

    t = np.linspace(0, duration, int(fs * duration), False)
    wave = np.sin(frequency * 2 * np.pi * t) * 0.3
    audio = (wave * 32767).astype(np.int16)

    sa.play_buffer(audio, 1, 2, fs).wait_done()




def tune_hyperparameters(X_train, y_train):

    grid = GridSearchCV(LogisticRegression(max_iter=200), PARAM_GRID,cv= CV_SPLITS)
    grid.fit(X_train, y_train)
    print("Best parameters:", grid.best_params_)
    print("Best cross-validation score:", grid.best_score_)
    return grid.best_estimator_



def error_beep():
    frequency = 1500  # Hz - higher pitch for "error"
    duration = 0.1    # seconds - short for snappy feel
    fs = 44100        # sample rate

    t = np.linspace(0, duration, int(fs * duration), False)
    wave = np.sin(frequency * 2 * np.pi * t) * 0.9  # louder
    audio = (wave * 32767).astype(np.int16)

    sa.play_buffer(audio, 1, 2, fs).wait_done()

