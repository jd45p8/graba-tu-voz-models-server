import math
import io
import os
import json
import base64
import requests

import numpy as np
import soundfile as sf

from tensorflow import keras
from librosa.feature import mfcc

URL_NOISE_REDUCE = os.environ['URL_NOISE_REDUCE']
WINDOW_LENGTH = int(os.environ['WINDOW_LENGTH'])
MODEL = keras.models.load_model('./model')

def predict(x):
    batch = np.full(shape=(1, x.shape[0], x.shape[1]), fill_value=0)
    batch[0] = x
    return MODEL.predict(batch)

def extract_features(samples, rate):
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

def lambda_handler(event):
    content = event['content']
    audioBytes = io.BytesIO(base64.b64decode(content))
    # Limpieza del audio
    print("Limpiando el audio")
    response_clean = requests.request('POST', URL_NOISE_REDUCE, files = {'file': ("audio.wav", audioBytes, "audio/wav")})
    audioCleanBytes = io.BytesIO(response_clean.content)
    # Extraer los samples del audio
    print("Extrayendo los samples del audio")
    samples, rate = sf.read(audioCleanBytes)
    # Extraer las características
    print("Extrayendo las características")
    x = extract_features(samples, rate)
    # Predecir
    print("Prediciendo")
    predicted = predict(x)
    return {
        'statusCode': 200,
        'body': json.dumps(str(predicted))
    }

if __name__ == "__main__":
    event = json.load(open('./audio.json'))
    lambda_handler(event, None)