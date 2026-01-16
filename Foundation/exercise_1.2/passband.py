import numpy as np

def generate_passband(I_baseband, Q_baseband, sampling_rate, duration, carrier_frequency):
    num_samples = sampling_rate * duration
    time_axis = np.linspace(0, duration, num_samples, endpoint=False)

    # Generate carrier signals
    carrier_I = np.cos(2 * np.pi * carrier_frequency * time_axis)
    carrier_Q = np.sin(2 * np.pi * carrier_frequency * time_axis)

    # Modulate baseband signals to passband
    passband_signal = I_baseband * carrier_I - Q_baseband * carrier_Q
    return time_axis, passband_signal