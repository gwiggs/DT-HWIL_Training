# Validate your Python setup
# Create a script that:
# - Generates a sine wave at 1 kHz sampled at 44.1 kHz
# - Plots time domain representation
# - Computes and plots FFT
# - Saves results to file
import numpy as np
import matplotlib.pyplot as plt

#CONSTANTS
SAMPLING_RATE = 44100  # in Hz
SIGNAL_FREQUENCY = 1000  # in Hz
DURATION = 1  # in seconds
AMPLITUDE = 1.0

#setup the sinewave parameters

num_samples = SAMPLING_RATE * DURATION

time_axis = np.linspace(0, DURATION, num_samples, endpoint=False)
sine_wave = AMPLITUDE * np.sin(2 * np.pi * SIGNAL_FREQUENCY * time_axis)

# Plot time domain representation
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time_axis[:1000], sine_wave[:1000])  # Plot first 1000 samples
plt.title('Time Domain Representation of 1 kHz Sine Wave')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()  
plt.show()

# Compute FFT
fft_result = np.fft.fft(sine_wave)
fft_magnitude = np.abs(fft_result) / num_samples
frequencies = np.fft.fftfreq(num_samples, 1 / SAMPLING_RATE)
half_n = num_samples // 2
fft_magnitude = fft_magnitude[:half_n]
frequencies = frequencies[:half_n]
# Plot FFT
plt.subplot(2, 1, 2)
plt.plot(frequencies, fft_magnitude)
plt.title('FFT of 1 kHz Sine Wave')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.xlim(0, 5000)  # Limit x-axis to 5 kHz for better visibility
plt.grid()
plt.tight_layout()
plt.show()
