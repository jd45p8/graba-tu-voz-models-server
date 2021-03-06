import prediction
import download_models
import os

import numpy as np

from flask import Flask, request, jsonify

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
PORT = int(os.environ.get("PORT", 8080))

ALLOWED_TYPES = ['audio/wav', 'audio/wave']
MAX_FILES = int(os.environ['MAX_FILES'])

def verify_files(files):
    '''
    Verifica si los archivos recibidos pueden ser procesado
    - files: diccionario de archivos recibidos
    Retorna: las llaves de los archivos si todos son permitidos o None
    '''
    file_keys = [key for key in files.keys()]
    allowed_keys = []
    for key in file_keys:
        if request.files[key].content_type in ALLOWED_TYPES:
            allowed_keys.append(key)
        else:
            return None
    return allowed_keys

@app.route('/speaker', methods=['POST'])
def speaker():
    '''
    Predice el hablante relacionado con la pista de audio recibida
    '''
    file_keys = verify_files(request.files)

    if not file_keys:
         return {
            "message": "No se envió ningún archivo o no es de un tipo no permitido."
        }, 422

    if len(file_keys) > MAX_FILES:
        return {
            "message": f"Se superó el máximo permitido({MAX_FILES})."
        }, 422
    
    updated = download_models.download_speaker()
       
    audio = request.files["file"].stream
    predicted = prediction.predict_speaker(audio, int(request.form["phraseSamples"]), updated)
    return jsonify(predicted), 200

@app.route('/character', methods=['POST'])
def character():
    '''
    Predice la etiqueta relacionada con la pista de audio recibida
    '''
    file_keys = verify_files(request.files)
    if not file_keys:
        return {
            "message": "No se enviaron archivos o no de un tipo no permitido."
        }, 422

    if len(file_keys) > MAX_FILES:
        return {
            "message": f"Se superó el máximo permitido({MAX_FILES})."
        }, 422
    
    updated = download_models.download_character()
        
    audio = request.files["file"].stream
    predicted = prediction.predict_character(audio, int(request.form["phraseSamples"]), updated)
    return jsonify(predicted), 200

@app.errorhandler(405)
def method_not_allowed(error):
    '''
    Responde a los errores de método no permitido en alguna dirección.
    '''
    return {
        "message": "El método no está permitido."
    }, 405

@app.errorhandler(404)
def page_no_found(error):
    '''
    Responde a los errores al solicitar un recurso y este no se encuentra disponible.
    '''
    return {
        "message": "Recurso no encontrado."
    }, 404

# @app.errorhandler(Exception)
# def something_went_wrong(error):
#     print(error)
#     '''
#     Responde a cualquier otro error como un error del servidor.
#     '''
#     return {
#         "message": "Algo ha salido mal."
#     }, 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)