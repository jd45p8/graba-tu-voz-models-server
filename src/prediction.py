import math
import io
import os
import json
import base64
import operator

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

SPEAKER_MODEL = None
SPEAKER_LABEL_ENCODER = None

CHARACTER_MODEL = None
CHARACTER_LABEL_ENCODER = None

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

def preprocessing(audioBytesIO, phrase_samples):
    '''
    Realiza el preprocesamiento necesario al audio recibido, solo se esperan 4 dígitos
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
    features_array = []
    start = 0
    end = 0
    for i in range(1, 4):
        start = end
        end = i*phrase_samples
        samplesPartition = cleanSamples[start:end]
        # Cortar audios
        samplesPower = np.power(samplesPartition, 2)
        samplesConvolution = np.convolve(a=samplesPower, v=np.ones(10000), mode='same')
        classFreq, classLim = np.histogram(samplesConvolution)
        selectedConvPowerSamples = samplesPartition[samplesConvolution > classLim[2]]
        # Extraer las características del audio
        features = extract_features(selectedConvPowerSamples, rate)
        features_array.append(features)
    
    samplesPartition = cleanSamples[end:-1]
    # Cortar audios
    samplesPower = np.power(samplesPartition, 2)
    samplesConvolution = np.convolve(a=samplesPower, v=np.ones(10000), mode='same')
    classFreq, classLim = np.histogram(samplesConvolution)
    selectedConvPowerSamples = samplesPartition[samplesConvolution > classLim[2]]
    # Extraer las características del audio
    features = extract_features(selectedConvPowerSamples, rate)
    features_array.append(features)

    return np.array(features_array)

def predict_speaker(audioFile, phrase_samples, updateModel):
    '''
    Estima a que hablante pertenece el audio en el archivo recibido
    - audioFile: archivo de audiorecibido en un BytesIO de la librería io
    - updateModel: determina si se deben volver a cargar los modelos
    '''
    global SPEAKER_MODEL, SPEAKER_LABEL_ENCODER

    if updateModel or not SPEAKER_MODEL:
        SPEAKER_MODEL = keras.models.load_model(f'{MODELS_PATH}/{SPEAKER_MODEL_KEY}')
        SPEAKER_LABEL_ENCODER = LabelEncoder()
        SPEAKER_LABEL_ENCODER.classes_ = np.load(
            f'{MODELS_PATH}/{SPEAKER_MODEL_LABELS_KEY}',
            allow_pickle=True)

    batch = preprocessing(audioFile, phrase_samples)
    # Predecir
    print('Prediciendo')
    probabilities = SPEAKER_MODEL.predict(batch)
    response = []

    # Se selecciona el top 5 con una votación
    score_dict = {}
    score_list = []
    for prob in probabilities:
        keys = prob.argsort()
        for index, key in enumerate(keys):
            if not key in score_dict:
                score_dict[key] = len(score_list)

                elem = [key, # Email
                    np.median(probabilities[:, key]), #Probabilidad
                    index * prob[key]] # Puntaje de la votación

                score_list.append(elem)
            else:
                pos = score_dict[key]
                score_list[pos][2] += index * prob[key]

    score_sorted = sorted(score_list, key=operator.itemgetter(2), reverse=True)
    top_5 = sorted(score_sorted[:5], key=operator.itemgetter(1), reverse=True)

    for key, probability, score in top_5:
        response.append({
            "label": SPEAKER_LABEL_ENCODER.inverse_transform([key])[0],
            "probability": float(probability)
        })
    
    return response

def predict_character(audioFile, phrase_samples, updateModel):
    '''
    Estima a que caracter corresponde el audio en el archivo de audio recibido
    - audioFile: archivo de audiorecibido en un BytesIO de la librería io
    - updateModel: determina si se deben volver a cargar los modelos
    '''
    global CHARACTER_MODEL, CHARACTER_LABEL_ENCODER

    if updateModel or not CHARACTER_MODEL:
        CHARACTER_MODEL = keras.models.load_model(f'{MODELS_PATH}/{CHARACTER_MODEL_KEY}')
        CHARACTER_LABEL_ENCODER = LabelEncoder()
        CHARACTER_LABEL_ENCODER.classes_ = np.load(
            f'{MODELS_PATH}/{CHARACTER_MODEL_LABELS_KEY}',
            allow_pickle=True)

    batch = preprocessing(audioFile, phrase_samples)
    # Predecir
    print('Prediciendo')
    probabilities =  CHARACTER_MODEL.predict(batch)
    response = []

    for prob in probabilities:
        label_index = np.argmax(prob)
        response.append({
            "label": CHARACTER_LABEL_ENCODER.inverse_transform([label_index])[0],
            "probability": float(prob[label_index])
        })
    
    return response