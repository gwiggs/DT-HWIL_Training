from matplotlib import pyplot as plt

def plot_time_domain(signal_type,time_axis, I_baseband, Q_baseband, passband_signal):
    plt.figure(figsize=(12, 8))

    # Plot I and Q baseband signals
    plt.subplot(3, 1, 1)
    plt.plot(time_axis[:1000], I_baseband[:1000], label='I Baseband')
    plt.plot(time_axis[:1000], Q_baseband[:1000], label='Q Baseband')
    plt.title(f'{signal_type} Baseband Signals (First 1000 samples)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()

    # Plot Passband signal
    plt.subplot(3, 1, 2)
    plt.plot(time_axis[:1000], passband_signal[:1000], color='orange')
    plt.title(f'{signal_type} Passband Signal (First 1000 samples)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid()

    plt.tight_layout()
    plt.show()
    
def plot_constellation(I_symbols, Q_symbols):
    plt.figure(figsize=(6, 6))
    plt.scatter(I_symbols, Q_symbols, color='red')
    plt.title('Constellation Diagram')
    plt.xlabel('In-Phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.grid()
    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='black', lw=0.5)
    plt.show()

def eye_diagram(signal, samples_per_symbol, num_symbols=5):
    plt.figure(figsize=(10, 6))
    for i in range(num_symbols):
        start_idx = i * samples_per_symbol
        end_idx = start_idx + samples_per_symbol
        plt.plot(signal[start_idx:end_idx], color='blue', alpha=0.5)
    plt.title('Eye Diagram')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()
    
def plat_spectrogram(signal, sampling_rate):
    plt.figure(figsize=(10, 6))
    plt.specgram(signal, NFFT=1024, Fs=sampling_rate, noverlap=512, cmap='plasma')
    plt.title('Spectrogram')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar(label='Intensity [dB]')
    plt.show()  