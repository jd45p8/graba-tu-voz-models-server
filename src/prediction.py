import math
import io
import os
import json
import base64

import numpy as np
import soundfile as sf
import noisereduce as nr

from tensorflow import keras
from librosa.feature import mfcc
from sklearn.preprocessing import LabelEncoder

WINDOW_LENGTH = int(os.environ['WINDOW_LENGTH'])
MODELS_PATH = os.environ['MODELS_PATH']
CHARACTER_MODEL_KEY = os.environ['CHARACTER_MODEL_KEY']
CHARACTER_MODEL_LABELS_KEY = os.environ['CHARACTER_MODEL_LABELS_KEY']
SPEAKER_MODEL_KEY = os.environ['SPEAKER_MODEL_KEY']
SPEAKER_MODEL_LABELS_KEY = os.environ['SPEAKER_MODEL_LABELS_KEY']

def extract_features(samples, rate):
    '''
    Extrae las catacterísticas necesarias para la predicción
    - samples: son los samples del audio
    - rate: la velocidad de muestreo del audio
    Retorna: las características del audio recibido para la predicción
    '''
    if samples.shape[0] <= WINDOW_LENGTH:
        # Rellenar audios cortos
        extra = (WINDOW_LENGTH - samples.shape[0])    
        r_zeros = np.zeros(math.ceil(extra/2))
        l_zeros = np.zeros(extra//2)
        windowSamples = np.concatenate((r_zeros, samples, l_zeros)) 
    else:
        startPos = (samples.shape[0] - WINDOW_LENGTH)//2
        windowSamples = samples[startPos : startPos + WINDOW_LENGTH]

    features = mfcc(y=windowSamples, sr=rate, n_mfcc=64, hop_length=1024)
    return features

def preprocessing(audioBytesIO):
    '''
    Realiza el preprocesamiento necesario al audio recibido
    - audioBytesIO: es el audio en un io.BytesIO
    Retorna: las características del audio recibido para su predicción
    '''
    # Extraer los samples del audio
    print('Extrayendo los samples del audio')
    samples, rate = sf.read(audioBytesIO)
    # Limpieza del audio
    print('Limpiando el audio')
    noisySamples = samples[-rate:-1]
    cleanSamples = nr.reduce_noise(audio_clip=samples, noise_clip=noisySamples)[:-rate]
    # Extraer las características
    print('Extrayendo las características')
    x = extract_features(cleanSamples, rate)
    return x

def predict_speaker(audioFile):
    '''
    Estima a que hablante pertenece el audio en el archivo recibido
    - audioFile: archivo de audiorecibido en un BytesIO de la librería io
    '''
    model = keras.models.load_model(f'{MODELS_PATH}/{SPEAKER_MODEL_KEY}')
    labelEncoder = LabelEncoder()
    labelEncoder.classes_ = np.load(f'{MODELS_PATH}/{SPEAKER_MODEL_LABELS_KEY}', allow_pickle=True)

    x = preprocessing(audioFile)
    batch = np.full(shape=(1, x.shape[0], x.shape[1]), fill_value=0)
    batch[0] = x
    # Predecir
    print('Prediciendo')
    probabilities =  model.predict(batch)[0]
    top5_indexes = np.argpartition(probabilities, -5)[-5:]
    response = []
    for index in top5_indexes:
        response.append({
            "label": labelEncoder.inverse_transform([index])[0],
            "probability": float(probabilities[index])
        })
    return response

def predict_character(audioFile):
    '''
    Estima a que caracter corresponde el audio en el archivo de audio recibido
    - audioFile: archivo de audiorecibido en un BytesIO de la librería io
    '''
    model = keras.models.load_model(f'{MODELS_PATH}/{CHARACTER_MODEL_KEY}')
    labelEncoder = LabelEncoder()
    labelEncoder.classes_ = np.load(f'{MODELS_PATH}/{CHARACTER_MODEL_LABELS_KEY}', allow_pickle=True)

    x = preprocessing(audioFile)
    batch = np.full(shape=(1, x.shape[0], x.shape[1]), fill_value=0)
    batch[0] = x
    # Predecir
    print('Prediciendo')
    probabilities =  model.predict(batch)[0]
    label_index = np.argmax(probabilities)
    return {
        "label": labelEncoder.inverse_transform([label_index])[0],
        "probability": float(probabilities[label_index])
    }