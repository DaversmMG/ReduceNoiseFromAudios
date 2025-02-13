import IPython.display as ipd
import matplotlib.pyplot as plt
import urllib.request
import numpy as np
import io
import soundfile as sf
import noisereduce as nr
from scipy.signal import butter, lfilter

# Función para aplicar un filtro de paso alto y mejorar la reducción de ruido
def highpass_filter(data, cutoff=100, fs=44100, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return lfilter(b, a, data)

# Cargar datos de audio desde la URL (Le podemos meter DataSets de audios)
url = "https://raw.githubusercontent.com/DaversmMG/audios/main/audiocall_original.WAV"
response = urllib.request.urlopen(url)
data, rate = sf.read(io.BytesIO(response.read()))

# Agregar ruido artificial para mejorar el contraste del rudio y poder reducirlo de forma mas eficiente.
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

# Guardar el audio procesado
nonstationary_filename = "reduced_noise_nonstationary.wav"
sf.write(nonstationary_filename, reduced_noise_nonstationary, rate)

print(f"✅ Archivo guardado: {nonstationary_filename}")

# Reproducir el audio con ruido añadido
print("\n🔊 Audio con Ruido:")
ipd.display(ipd.Audio(data=audio_clip_cafe, rate=rate))

# Reproducir el audio después de la reducción de ruido (Non-Stationary)
print("\n🔊 Audio después de reducción de ruido:")
ipd.display(ipd.Audio(data=reduced_noise_nonstationary, rate=rate))
