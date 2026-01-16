import numpy as np

def generate_bpsk_I_Q_baseband(sampling_rate, symbol_rate, duration):
    # Generate random binary data
    num_symbols = symbol_rate * duration
    binary_data = np.random.randint(0, 2, num_symbols)
    
    # BPSK mapping (0 -> -1, 1 -> +1)
    symbols = np.zeros(num_symbols)
    for n in range(num_symbols):
        if binary_data[n] == 0:
            symbols[n] = -1
        else:
            symbols[n] = 1
    
    # uposample symbols to match sampling rate
    samples_per_symbol = sampling_rate // symbol_rate
    num_samples = sampling_rate * duration

    I_baseband = np.zeros(num_samples)
    Q_baseband = np.zeros(num_samples)
    for n in range(num_symbols):
        start_idx = n * samples_per_symbol
        end_idx = (n+1) * samples_per_symbol
        for k in range(start_idx, end_idx):
            I_baseband[k] = symbols[n]
            Q_baseband[k] = 0  # BPSK has no Q component
    return I_baseband, Q_baseband
    
