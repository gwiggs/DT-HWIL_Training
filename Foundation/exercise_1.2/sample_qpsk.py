import numpy as np

def generate_qpsk_I_Q_baseband(sampling_rate, symbol_rate, duration):
    # Generate random binary data
    num_symbols = symbol_rate * duration
    binary_data = np.random.randint(0, 2, num_symbols * 2)  # 2 bits per QPSK symbol
    
    # QPSK mapping
    symbols_I = np.zeros(num_symbols)
    symbols_Q = np.zeros(num_symbols)
    
    for n in range(num_symbols):
        bit1 = binary_data[2*n]
        bit2 = binary_data[2*n + 1]
        
        if bit1 == 0 and bit2 == 0:
            symbols_I[n] = 1 / np.sqrt(2)
            symbols_Q[n] = 1 / np.sqrt(2)
        elif bit1 == 0 and bit2 == 1:
            symbols_I[n] = -1 / np.sqrt(2)
            symbols_Q[n] = 1 / np.sqrt(2)
        elif bit1 == 1 and bit2 == 1:
            symbols_I[n] = -1 / np.sqrt(2)
            symbols_Q[n] = -1 / np.sqrt(2)
        elif bit1 == 1 and bit2 == 0:
            symbols_I[n] = 1 / np.sqrt(2)
            symbols_Q[n] = -1 / np.sqrt(2)
    
    # Upsample symbols to match sampling rate
    samples_per_symbol = sampling_rate // symbol_rate
    num_samples = sampling_rate * duration

    I_baseband = np.zeros(num_samples)
    Q_baseband = np.zeros(num_samples)
    
    for n in range(num_symbols):
        start_idx = n * samples_per_symbol
        end_idx = (n + 1) * samples_per_symbol
        for k in range(start_idx, end_idx):
            I_baseband[k] = symbols_I[n]
            Q_baseband[k] = symbols_Q[n]
    
    return I_baseband, Q_baseband