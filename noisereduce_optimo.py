import IPython.display as ipd
import matplotlib.pyplot as plt
import urllib.request
import numpy as np
import io
import soundfile as sf
import noisereduce as nr
import librosa
from scipy.signal import butter, lfilter

# Función para aplicar un filtro de paso alto y mejorar la reducción de ruido
def highpass_filter(data, cutoff=100, fs=44100, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return lfilter(b, a, data)

# Función para eliminar silencios basados en la energía del audio
def remove_silence(audio, rate, top_db=25):
    """
    Elimina segmentos silenciosos o con ruido de fondo donde no hay voz.
    
    Parámetros:
        audio (numpy array): Señal de audio.
        rate (int): Frecuencia de muestreo.
        top_db (int): Umbral en decibeles para detectar silencio.
    
    Retorna:
        audio_recortado (numpy array): Audio sin los segmentos silenciosos.
    """
    # Detectar segmentos donde hay voz
    intervals = librosa.effects.split(audio, top_db=top_db)

    # Concatenar los segmentos detectados para eliminar los silencios
    audio_recortado = np.concatenate([audio[start:end] for start, end in intervals])

    return audio_recortado

# Cargar datos de audio desde la URL (Le podemos meter DataSets de audios)
url = "https://raw.githubusercontent.com/DaversmMG/audios/main/audiocall_original.WAV"
response = urllib.request.urlopen(url)
data, rate = sf.read(io.BytesIO(response.read()))

# Agregar ruido artificial para mejorar el contraste del ruido y poder reducirlo de forma más eficiente.
snr = 2  # Relación señal a ruido
noise_clip = data / snr
audio_clip_cafe = data + noise_clip

# Aplicar filtro de paso alto antes de la reducción de ruido
filtered_audio = highpass_filter(audio_clip_cafe, fs=rate)

# Gráfico del audio original vs. con ruido
plt.figure(figsize=(20, 4))
plt.plot(data, label="Audio Original", alpha=0.5, color="blue")
plt.plot(audio_clip_cafe, label="Audio con Ruido", alpha=0.7, color="red")
plt.legend()
plt.title("Comparación de Audio Original vs. Audio con Ruido")
plt.xlabel("Tiempo (muestras)")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()

# **Non-stationary noise reduction - SCRIPT DE AUDIO DESEADO**
reduced_noise_nonstationary = nr.reduce_noise(y=filtered_audio, sr=rate, thresh_n_mult_nonstationary=2, stationary=False)

# Gráfico comparativo de audio con ruido y audio reducido (Non-Stationary)
plt.figure(figsize=(20, 4))
plt.plot(audio_clip_cafe, label="Audio con Ruido", alpha=0.5, color="skyblue")
plt.plot(reduced_noise_nonstationary, label="Ruido Reducido (Non-Stationary)", alpha=0.9, color="green")
plt.legend()
plt.title("Comparación de Audio con Ruido vs. Ruido Reducido")
plt.xlabel("Tiempo (muestras)")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()

# Eliminar los silencios después de la reducción de ruido
audio_cleaned = remove_silence(reduced_noise_nonstationary, rate, top_db=25)

# Graficar la comparación de audio antes y después de eliminar silencios
plt.figure(figsize=(20, 4))
plt.plot(reduced_noise_nonstationary, label="Ruido Reducido", alpha=0.5, color="green")
plt.plot(audio_cleaned, label="Ruido Reducido sin Silencios", alpha=0.9, color="purple")
plt.legend()
plt.title("Comparación de Audio Reducido vs. Audio sin Silencios")
plt.xlabel("Tiempo (muestras)")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()

# Guardar los archivos procesados
final_filename = "reduced_noise_no_silence.wav"
sf.write(final_filename, audio_cleaned, rate)

print(f"Archivo guardado: {final_filename}")

