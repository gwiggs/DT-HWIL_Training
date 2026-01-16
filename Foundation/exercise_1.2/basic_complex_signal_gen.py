import numpy as np
import matplotlib.pyplot as plt

# Constants:
SAMPLING_RATE = 44100  # in Hz
DURATION = 1  # in seconds
AMPLITUDE = 1.0
SIGNAL_FREQUENCY = 1000  # in Hz

def generate_complex_exponential_signal(phase_shift=0):
    # create the time axis
    num_samples = SAMPLING_RATE * DURATION
    time_axis = np.linspace(0, DURATION, num_samples, endpoint=False)

    # Generate complex exponential signal using Euler's formula
    omega = 2 * np.pi * SIGNAL_FREQUENCY
    complex_signal = np.zeros(num_samples, dtype=complex)
    for n in range(num_samples):
        t = time_axis[n]
        theta = omega * t + phase_shift
        real_part = AMPLITUDE * np.cos(theta)
        imag_part = AMPLITUDE * np.sin(theta)
        complex_signal[n] = real_part + 1j * imag_part
    return time_axis, complex_signal



