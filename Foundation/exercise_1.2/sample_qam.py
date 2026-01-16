import numpy as np
import math
def generate_qam_constellation(M):
    """ Generate QAM constellation points for M-QAM """
    """M must be a perfect square"""
    m_side = int(math.sqrt(M))
    if m_side != math.floor(m_side):
        raise ValueError("M must be a perfect square (4, 16, 64, 256 etc.)")
    """Generate amplittude levels"""
    amplitude_levels = np.zeros(m_side)
    for i in range(m_side):
        amplitude_levels[i] = - (m_side - 1) + 2 * i
    """Generate constellation points"""
    constellation = np.zeros((M, 2))  # Each row is a point (I, Q)
    index = 0
    for i in range(m_side):
        for q in range(m_side):
            constellation[index, 0] = amplitude_levels[i]  # I component
            constellation[index, 1] = amplitude_levels[q]  # Q component
            index += 1
    """Normalise constellation for unit average power"""
    avg_power = np.mean(np.sum(constellation**2, axis=1))
    constellation /= np.sqrt(avg_power)
    return constellation

def create_qam_bit_mapping(M, mapping_type):
    """Create QAM bit mapping"""
    """ mapping type can be 'binary' or 'gray' """
    bits_per_symbol = int(math.log2(M))
    if bits_per_symbol != math.floor(bits_per_symbol):
        raise ValueError("M must be a power of 2 (4, 16, 64, 256 etc.)")
    constellation, _ = generate_qam_constellation(M)
    grid_size = int(math.sqrt(M))
    bits_per_dimension = bits_per_symbol // 2
    """Create mapping dictionary"""
    bit_to_symbol = {}
    symbol_to_bit = {}
    
    for i in range(M):
        I_index = math.floor(i / grid_size)
        Q_index = i % grid_size
        if mapping_type == "gray":
            I_bits = binary_to_gray(I_index, bits_per_dimension)
            Q_bits = binary_to_gray(Q_index, bits_per_dimension)
        else:
            I_bits = decimal_to_binary(I_index, bits_per_dimension)
            Q_bits = decimal_to_binary(Q_index, bits_per_dimension)
        bit_pattern = I_bits + Q_bits
        bit_to_symbol[bit_pattern] = constellation[i]
        symbol_to_bit[i] = bit_pattern
        
    return bit_to_symbol, symbol_to_bit

def binary_to_gray(n, num_bits):
    gray_code = n ^ (n >> 1)
    return decimal_to_binary(gray_code, num_bits)

def decimal_to_binary(n, num_bits):
    bits = np.zeros(num_bits)
    for k in range (num_bits, 0, -1):
        bits[k] = n % 2
        n = n // 2
    return bits

def binary_to_decimal(bits):
    n = 0
    num_bits = len(bits)
    for k in range(num_bits):
        n += bits[num_bits - k - 1] * (2 ** k)
    return n

def generate_qam_I_Q_baseband(M, num_symbols, smaples_per_symbol, pulse_shape_type, pulse_shape_params):
    """ Generate QAM baseband I and Q signals """
    """ M: Modulation order (4, 16, 64, etc.) """
    """ num_symbols: Number of symbols to generate """
    """ samples_per_symbol: Number of samples per symbol """
    """ pulse_shape_type: 'rectangular' or 'raised_cosine' """
    """ pulse_shape_params: parameters for pulse shaping (e.g., roll-off factor for raised cosine) """
    pass