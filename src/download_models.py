import os
import boto3
import zipfile
import shutil

from pathlib import Path

MODELS_PATH = os.environ['MODELS_PATH']
AWS_ACCESS_KEY_ID = os.environ['AWS_ACCESS_KEY_ID']
AWS_SECRET_KEY_ID = os.environ['AWS_SECRET_KEY_ID']
MODELS_BUCKET_NAME = os.environ['MODELS_BUCKET_NAME']
CHARACTER_MODEL_KEY = os.environ['CHARACTER_MODEL_KEY']
CHARACTER_MODEL_LABELS_KEY = os.environ['CHARACTER_MODEL_LABELS_KEY']
SPEAKER_MODEL_KEY = os.environ['SPEAKER_MODEL_KEY']
SPEAKER_MODEL_LABELS_KEY = os.environ['SPEAKER_MODEL_LABELS_KEY']

def download_all():
    '''
    Descarga los modelos del S3 Bucket indicado en las variables de entorno.
    '''
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_KEY_ID
    )
    s3_resource = session.resource('s3')
    s3_grabatuvozmodels = s3_resource.Bucket('grabatuvozmodels')

    download_model_file(CHARACTER_MODEL_KEY, s3_grabatuvozmodels)
    download_model_file(SPEAKER_MODEL_KEY, s3_grabatuvozmodels)
    download_model_file(CHARACTER_MODEL_LABELS_KEY, s3_grabatuvozmodels)
    download_model_file(SPEAKER_MODEL_LABELS_KEY, s3_grabatuvozmodels)

def download_speaker():
    '''
    Descarga los modelos del hablante del S3 Bucket indicado en las variables de entorno.
    '''
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_KEY_ID
    )
    s3_resource = session.resource('s3')
    s3_grabatuvozmodels = s3_resource.Bucket('grabatuvozmodels')

    download_model_file(SPEAKER_MODEL_KEY, s3_grabatuvozmodels)
    download_model_file(SPEAKER_MODEL_LABELS_KEY, s3_grabatuvozmodels)

def download_character():
    '''
    Descarga los modelos del S3 Bucket indicado en las variables de entorno.
    '''
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_KEY_ID
    )
    s3_resource = session.resource('s3')
    s3_grabatuvozmodels = s3_resource.Bucket('grabatuvozmodels')

    download_model_file(CHARACTER_MODEL_KEY, s3_grabatuvozmodels)
    download_model_file(CHARACTER_MODEL_LABELS_KEY, s3_grabatuvozmodels)

def download_model_file(key, bucket):
    '''
    Descarga el archivo de la llave indicada del S3 bucket
    - key la llave del archivo
    - bucket el objeto S3 Bucket que tiene el archivo
    '''
    etag_path = Path(f'{MODELS_PATH}/{key}_etag')
    final_path = Path(f'{MODELS_PATH}/{key}')
    temp_path = Path(f'/tmp/{key}')

    # Lee el etag del archivo si existe
    currentTag = None
    if etag_path.is_file():
        with open(etag_path,'r') as textIO:
            currentTag = textIO.read()
    
    obj = bucket.Object(key=key)
    # Finaliza la ejecucion de la función si no ha cambiado el archivo
    if currentTag == obj.e_tag:
        return    
    
    # Descarga el archivo nuevo
    obj.download_file(Filename=str(temp_path.resolve()))    
    # Guarda el nuevo etag
    with open(etag_path,'w') as textIO:
        textIO.write(obj.e_tag)
    # Si no es un .zip lo mueve a la capeta permanente
    if obj.content_type != 'application/zip':
        shutil.move(temp_path, final_path)
        return
    
    # Si es un .zip se extrae
    with zipfile.ZipFile(file=temp_path, mode='r') as zip_ref:
        if final_path.is_dir():
            # Borra la carpeta y su contenido     
            shutil.rmtree(path=final_path)
        final_path.mkdir()
        # Extrae el archivo .zip
        zip_ref.extractall(path=final_path)
    
    # Borra el archivo después de hacer la extracción
    temp_path.unlink()