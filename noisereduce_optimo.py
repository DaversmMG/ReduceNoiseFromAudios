import os
import subprocess
import numpy as np
import soundfile as sf
import noisereduce as nr
import librosa
import boto3 
from botocore.exceptions import NoCredentialsError
from scipy.signal import butter, lfilter
from dotenv import load_dotenv

load_dotenv()

# Configuración de AWS S3
S3_BUCKET_NAME = "sst-files-history"  # Reemplaza con el nombre de tu bucket
S3_FOLDER = "historial_audios/"      # Carpeta dentro del bucket

# Obtener credenciales de AWS desde .env
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Inicializar cliente S3 con credenciales cargadas
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

def upload_to_s3(file_path, bucket_name, s3_key):
    """Sube un archivo a AWS S3"""
    try:
        s3.upload_file(file_path, bucket_name, s3_key)
        print(f"✅ Archivo subido a S3: s3://{bucket_name}/{s3_key}")
    except NoCredentialsError:
        print("❌ ERROR: No se encontraron credenciales de AWS en el .env.")

# Función para aplicar un filtro de paso alto y mejorar la reducción de ruido
def highpass_filter(data, cutoff=100, fs=44100, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return lfilter(b, a, data)

# Función para eliminar silencios basados en la energía del audio
def remove_silence(audio, rate, top_db=25):
    intervals = librosa.effects.split(audio, top_db=top_db)
    audio_recortado = np.concatenate([audio[start:end] for start, end in intervals])
    return audio_recortado

# Función para aplicar un filtro de ecualización para mejorar la claridad de la voz
def equalize_voice(audio, rate):
    highpass_audio = highpass_filter(audio, cutoff=80, fs=rate)
    nyq = 0.5 * rate
    normal_cutoff_low = 3000 / nyq
    b, a = butter(2, normal_cutoff_low, btype='low', analog=False)
    lowpass_audio = lfilter(b, a, highpass_audio)
    return lowpass_audio

# Función para normalizar el audio
def normalize_audio(audio):
    return audio / np.max(np.abs(audio))

def final_audio_touch(input_path, output_path):
    """Aplica FFmpeg con filtros mejorados para mejorar la claridad de la voz sin cortar palabras."""
    command = [
        "ffmpeg", "-y", "-i", input_path,

        # Filtros de audio optimizados
        "-af",
        (
            "loudnorm=I=-16:TP=-1.5:LRA=11, "  # Normaliza el volumen manteniendo rango dinámico
            "dynaudnorm=f=150:g=5, "           # Reduce fluctuaciones de volumen sin alterar el inicio
            "afftdn=nf=-25, "                  # Reduce ruido ambiental sin cortar palabras
            "highpass=f=50, "                  # Mantiene frecuencias bajas importantes para la voz
            "lowpass=f=7500, "                 # Evita eliminar detalles agudos esenciales
            "compand=0.5|0.9:1.0|1.0:-80/-50/-20/-10/-5/0, "  # Evita cortes bruscos al inicio del habla
            "silenceremove=1:0:-45dB"          # Elimina ruido de fondo sin cortar sílabas iniciales
        ),

        # Configuración de salida en alta calidad
        "-ar", "44100",  # Mantiene el audio a 44.1 kHz
        "-ac", "2",      # Convierte a estéreo para mejor compatibilidad
        "-b:a", "192k",  # Bitrate de 192 kbps para mejor calidad

        output_path  # Archivo de salida
    ]

    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        print(f"✅ FFmpeg optimizó correctamente: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR en FFmpeg para {input_path}:")
        print(e.stderr)
        return None

    return output_path




# Función principal para procesar los audios en lote
def process_audio(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".wav"): 
            input_path = os.path.join(input_folder, filename)
            filename_base, ext = os.path.splitext(filename)  # Obtener nombre base sin extensión
            output_final = os.path.join(output_folder, f"{filename_base}_out{ext}")  # Agregar "_out.wav" al final

            print(f"\n Procesando: {filename}...")

            # Cargar el archivo de audio
            data, rate = sf.read(input_path)

            # Aplicar filtros personalizados
            noise_clip = data / 2  
            audio_clip_cafe = data + noise_clip
            filtered_audio = highpass_filter(audio_clip_cafe, fs=rate)
            reduced_noise_nonstationary = nr.reduce_noise(y=filtered_audio, sr=rate, thresh_n_mult_nonstationary=2, stationary=False)
            audio_cleaned = remove_silence(reduced_noise_nonstationary, rate, top_db=25)
            equalized_audio = equalize_voice(audio_cleaned, rate)
            normalized_audio = normalize_audio(equalized_audio)

            # Guardar audio temporal antes de FFmpeg
            temp_cleaned = os.path.join(output_folder, f"temp_{filename}")
            sf.write(temp_cleaned, normalized_audio, rate)
            print(f"Audio limpio guardado temporalmente: {temp_cleaned}")

            # Aplicar FFmpeg y guardar solo el archivo final
            final_audio = final_audio_touch(temp_cleaned, output_final)
            if final_audio:
                print(f" Audio final optimizado guardado: {final_audio}")
                # Subir el archivo final a S3
                s3_key = f"{S3_FOLDER}{filename_base}_out{ext}"
                upload_to_s3(final_audio, S3_BUCKET_NAME, s3_key)

            # Eliminar el archivo temporal después de la optimización
            os.remove(temp_cleaned)
            print(f" Archivo temporal eliminado: {temp_cleaned}")

# Definir carpetas de entrada y salida
input_folder = "./input_audios"  
output_folder = "./output_audios"  

# Ejecutar el procesamiento por lotes
process_audio(input_folder, output_folder)

print("\n✅✅✅ Procesamiento de audios completado. Solo los archivos finales se han guardado.")
