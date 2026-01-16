# QAM Generator

## Generic M-QAM COnstellation Generator
**"M" must be a perfect square number, ie. 16, 64, 256 etc**
```
function generate_qam_constellation(M):
    // Generate M-QAM constellation points
    // M must be a perfect square (4, 16, 64, 256, etc.)
    
    if not is_perfect_square(M):
        error("M must be a perfect square (4, 16, 64, 256, ...)")
    
    // Determine grid size (points per dimension)
    grid_size = sqrt(M)
    
    // For QAM, grid_size should be even for symmetric constellation
    // Exception: 4-QAM (QPSK) works with grid_size=2
    
    // Generate amplitude levels for each dimension
    // Levels are centered around zero: ..., -3, -1, +1, +3, ...
    levels = zeros(grid_size)
    
    for k = 0 to grid_size-1:
        // Map index to amplitude level
        // For grid_size = 4: levels = [-3, -1, +1, +3]
        // For grid_size = 8: levels = [-7, -5, -3, -1, +1, +3, +5, +7]
        levels[k] = 2*k - (grid_size - 1)
    end for
    
    // Create constellation by combining I and Q levels
    constellation = zeros(M) as complex array
    index = 0
    
    for i = 0 to grid_size-1:
        for q = 0 to grid_size-1:
            I_level = levels[i]
            Q_level = levels[q]
            
            constellation[index] = I_level + j * Q_level
            index = index + 1
        end for
    end for
    
    // Normalize constellation for unit average power
    avg_power = mean(abs(constellation)^2)
    normalization_factor = sqrt(avg_power)
    constellation = constellation / normalization_factor
    
    return constellation, normalization_factor
end function

function is_perfect_square(n):
    root = sqrt(n)
    return (root == floor(root))
end function
```
## Calculate Normalisation Factor Analytically
```
function calculate_qam_normalization_factor(M):
    // Analytical calculation of average power for M-QAM
    // Useful for understanding scaling without generating constellation
    
    grid_size = sqrt(M)
    
    // Average power calculation:
    // For square QAM, each dimension contributes equally
    // Sum of squares: 1² + 3² + 5² + ... + (grid_size-1)²
    
    sum_of_squares = 0
    for k = 0 to grid_size-1:
        level = 2*k - (grid_size - 1)
        sum_of_squares = sum_of_squares + level^2
    end for
    
    // Average power per dimension
    avg_power_per_dim = sum_of_squares / grid_size
    
    // Total average power (I and Q are independent)
    avg_power_total = 2 * avg_power_per_dim
    
    // Normalization factor
    normalization = sqrt(avg_power_total)
    
    return normalization
    
    // Closed form: normalization = sqrt(2*(M-1)/3)
end function

// Verification function
function verify_normalization():
    // Check that analytical and numerical match
    M_values = [4, 16, 64, 256]
    
    for M in M_values:
        constellation, norm_numerical = generate_qam_constellation(M)
        norm_analytical = calculate_qam_normalization_factor(M)
        
        print("M = " + M)
        print("  Numerical norm: " + norm_numerical)
        print("  Analytical norm: " + norm_analytical)
        print("  Avg power: " + mean(abs(constellation)^2))
        print("")
end function
```
## Bit Mapping for M-QAM
```
function create_qam_bit_mapping(M, mapping_type):
    // Create mapping between bit patterns and constellation points
    // mapping_type: "natural" or "gray"
    
    bits_per_symbol = log2(M)
    if bits_per_symbol != floor(bits_per_symbol):
        error("M must be power of 2")
    
    constellation, _ = generate_qam_constellation(M)
    grid_size = sqrt(M)
    bits_per_dimension = bits_per_symbol / 2
    
    // Create mapping dictionary
    bit_to_symbol = dictionary()
    symbol_to_bit = dictionary()
    
    for symbol_index = 0 to M-1:
        // Extract I and Q indices
        I_index = floor(symbol_index / grid_size)
        Q_index = symbol_index mod grid_size
        
        // Convert indices to bits
        if mapping_type == "gray":
            I_bits = binary_to_gray(I_index, bits_per_dimension)
            Q_bits = binary_to_gray(Q_index, bits_per_dimension)
        else:  // natural binary
            I_bits = decimal_to_binary(I_index, bits_per_dimension)
            Q_bits = decimal_to_binary(Q_index, bits_per_dimension)
        
        // Combine I and Q bits
        bit_pattern = concatenate(I_bits, Q_bits)
        
        // Store mapping
        bit_to_symbol[bit_pattern] = constellation[symbol_index]
        symbol_to_bit[symbol_index] = bit_pattern
    
    return bit_to_symbol, symbol_to_bit
end function

function binary_to_gray(decimal, num_bits):
    // Convert decimal to Gray code
    gray = decimal XOR (decimal >> 1)
    return decimal_to_binary(gray, num_bits)
end function

function decimal_to_binary(decimal, num_bits):
    // Convert decimal to binary array
    bits = zeros(num_bits)
    
    for k = num_bits-1 down to 0:
        bits[k] = decimal mod 2
        decimal = floor(decimal / 2)
    
    return bits
end function

function binary_to_decimal(bits):
    // Convert binary array to decimal
    decimal = 0
    
    for k = 0 to length(bits)-1:
        decimal = decimal * 2 + bits[k]
    
    return decimal
end function
```
## Complete Sample M-QAM Signal Generator
```
function generate_qam_baseband(M, num_symbols, samples_per_symbol, 
                               pulse_shape_type, pulse_shape_params):
    // Generate complete M-QAM baseband signal
    // M: constellation size (4, 16, 64, 256, etc.)
    // num_symbols: number of symbols to generate
    // samples_per_symbol: upsampling factor
    // pulse_shape_type: "rectangular", "rrc", "rc"
    // pulse_shape_params: dict with rolloff factor, span, etc.
    
    print("Generating " + M + "-QAM baseband signal...")
    
    // Step 1: Generate constellation
    constellation, normalization = generate_qam_constellation(M)
    print("  Constellation generated, normalization = " + normalization)
    
    // Step 2: Create bit mapping
    bit_to_symbol, symbol_to_bit = create_qam_bit_mapping(M, "gray")
    
    // Step 3: Generate random data
    bits_per_symbol = log2(M)
    total_bits = num_symbols * bits_per_symbol
    binary_data = random_binary(total_bits)
    print("  Generated " + total_bits + " random bits")
    
    // Step 4: Map bits to symbols
    symbols_I = zeros(num_symbols)
    symbols_Q = zeros(num_symbols)
    bit_patterns = empty_list()
    
    for n = 0 to num_symbols-1:
        // Extract bits for this symbol
        start_bit = n * bits_per_symbol
        end_bit = start_bit + bits_per_symbol
        symbol_bits = binary_data[start_bit:end_bit]
        
        // Look up constellation point
        complex_symbol = bit_to_symbol[symbol_bits]
        
        symbols_I[n] = real(complex_symbol)
        symbols_Q[n] = imag(complex_symbol)
        bit_patterns.append(symbol_bits)
    end for
    
    print("  Mapped bits to " + num_symbols + " symbols")
    
    // Step 5: Upsample symbols
    num_samples = num_symbols * samples_per_symbol
    I_upsampled = zeros(num_samples)
    Q_upsampled = zeros(num_samples)
    
    for n = 0 to num_symbols-1:
        // Place symbol at beginning of each symbol period
        index = n * samples_per_symbol
        I_upsampled[index] = symbols_I[n]
        Q_upsampled[index] = symbols_Q[n]
    end for
    
    // Step 6: Apply pulse shaping
    if pulse_shape_type == "rectangular":
        // Just hold values (no filtering needed for rectangular)
        I_baseband = zeros(num_samples)
        Q_baseband = zeros(num_samples)
        
        for n = 0 to num_symbols-1:
            start = n * samples_per_symbol
            end = start + samples_per_symbol
            I_baseband[start:end] = symbols_I[n]
            Q_baseband[start:end] = symbols_Q[n]
        end for
        
        pulse_filter = [1]  // No actual filter
    
    else if pulse_shape_type == "rrc":
        // Root-raised cosine
        rolloff = pulse_shape_params["rolloff"]  // beta
        span = pulse_shape_params["span"]  // filter span in symbols
        
        pulse_filter = root_raised_cosine_filter(rolloff, samples_per_symbol, span)
        
        I_baseband = convolve(I_upsampled, pulse_filter)
        Q_baseband = convolve(Q_upsampled, pulse_filter)
    
    else if pulse_shape_type == "rc":
        // Raised cosine
        rolloff = pulse_shape_params["rolloff"]
        span = pulse_shape_params["span"]
        
        pulse_filter = raised_cosine_filter(rolloff, samples_per_symbol, span)
        
        I_baseband = convolve(I_upsampled, pulse_filter)
        Q_baseband = convolve(Q_upsampled, pulse_filter)
    
    else:
        error("Unknown pulse shape type: " + pulse_shape_type)
    
    print("  Applied " + pulse_shape_type + " pulse shaping")
    
    // Create complex baseband signal
    complex_baseband = I_baseband + j * Q_baseband
    
    // Return comprehensive structure
    return {
        I_baseband: I_baseband,
        Q_baseband: Q_baseband,
        complex_baseband: complex_baseband,
        symbols_I: symbols_I,
        symbols_Q: symbols_Q,
        binary_data: binary_data,
        bit_patterns: bit_patterns,
        constellation: constellation,
        normalization: normalization,
        pulse_filter: pulse_filter,
        samples_per_symbol: samples_per_symbol,
        M: M
    }
end function
```
## Raised Cosine Filter (complement to RRC)
```
function raised_cosine_filter(beta, samples_per_symbol, num_symbols):
    // Raised cosine pulse shaping filter
    // beta = rolloff factor (0 to 1)
    // num_symbols = filter span in symbols
    
    num_taps = num_symbols * samples_per_symbol
    if num_taps mod 2 == 0:
        num_taps = num_taps + 1  // Make odd
    
    filter = zeros(num_taps)
    center = (num_taps - 1) / 2
    
    for n = 0 to num_taps-1:
        t = (n - center) / samples_per_symbol
        
        // Handle special cases
        if t == 0:
            // t = 0
            filter[n] = 1.0
        
        else if beta != 0 and abs(t) == 1/(2*beta):
            // t = ±1/(2*beta)
            filter[n] = (π/4) * sinc(1/(2*beta))
        
        else:
            // General case
            numerator = sin(π * t)
            denominator = π * t * (1 - (2*beta*t)^2)
            
            if abs(denominator) > 1e-10:
                filter[n] = (numerator / (π*t)) * cos(π*beta*t) / (1 - (2*beta*t)^2)
            else:
                filter[n] = 0
    
    // Normalize for unit energy
    energy = sqrt(sum(filter^2))
    filter = filter / energy
    
    return filter
end function

function sinc(x):
    // Sinc function: sin(πx)/(πx)
    if abs(x) < 1e-10:
        return 1.0
    else:
        return sin(π * x) / (π * x)
end function
```
## Example Usage Functions
```
function example_16qam():
    // Example: Generate 16-QAM signal
    
    M = 16
    num_symbols = 1000
    samples_per_symbol = 8
    
    pulse_params = {
        "rolloff": 0.35,
        "span": 10
    }
    
    signal = generate_qam_baseband(M, num_symbols, samples_per_symbol, 
                                   "rrc", pulse_params)
    
    print("Signal generated:")
    print("  Total samples: " + length(signal.I_baseband))
    print("  Avg power: " + mean(abs(signal.complex_baseband)^2))
    
    return signal
end function

function example_64qam():
    // Example: Generate 64-QAM signal
    
    M = 64
    num_symbols = 500
    samples_per_symbol = 10
    
    pulse_params = {
        "rolloff": 0.25,
        "span": 8
    }
    
    signal = generate_qam_baseband(M, num_symbols, samples_per_symbol, 
                                   "rrc", pulse_params)
    
    return signal
end function

function example_256qam():
    // Example: Generate 256-QAM signal
    
    M = 256
    num_symbols = 200
    samples_per_symbol = 8
    
    pulse_params = {
        "rolloff": 0.5,
        "span": 12
    }
    
    signal = generate_qam_baseband(M, num_symbols, samples_per_symbol, 
                                   "rrc", pulse_params)
    
    return signal
end function
```
## Visualisation Helper Function
```
function visualize_constellation_properties(M):
    // Visualize constellation and print key properties
    
    constellation, normalization = generate_qam_constellation(M)
    grid_size = sqrt(M)
    
    create_figure()
    
    // Plot constellation points
    scatter(real(constellation), imag(constellation), 
            s=100, color='blue', marker='o', label='Constellation Points')
    
    // Draw decision boundaries
    for k = 1 to grid_size-1:
        // Horizontal boundaries
        boundary_level = (constellation[k*grid_size].imag + 
                         constellation[(k-1)*grid_size].imag) / 2
        plot_horizontal_line(y=boundary_level, color='gray', 
                           linestyle='--', alpha=0.5)
        
        // Vertical boundaries
        boundary_level = (constellation[k].real + 
                         constellation[k-1].real) / 2
        plot_vertical_line(x=boundary_level, color='gray', 
                          linestyle='--', alpha=0.5)
    end for
    
    // Add origin
    scatter([0], [0], s=50, color='red', marker='x', label='Origin')
    
    xlabel("I (In-phase)")
    ylabel("Q (Quadrature)")
    title(M + "-QAM Constellation")
    axis_equal()
    grid_on()
    legend()
    
    // Print properties
    print(M + "-QAM Properties:")
    print("  Grid size: " + grid_size + " x " + grid_size)
    print("  Bits per symbol: " + log2(M))
    print("  Normalization factor: " + normalization)
    print("  Average power: " + mean(abs(constellation)^2))
    print("  Peak power: " + max(abs(constellation)^2))
    print("  PAPR: " + (max(abs(constellation)^2) / mean(abs(constellation)^2)))
    
    // Minimum distance (between adjacent points)
    min_distance = infinity
    for i = 0 to M-1:
        for j = i+1 to M-1:
            distance = abs(constellation[i] - constellation[j])
            if distance > 1e-10 and distance < min_distance:
                min_distance = distance
    
    print("  Minimum distance: " + min_distance)
    
    return constellation
end function
```
## Advanced: Non-Square Constellation
```
function generate_rectangular_qam(M):
    // Handle non-square QAM (e.g., 32-QAM, 128-QAM)
    // These use rectangular grids
    
    if M == 32:
        // 32-QAM: 4x8 grid
        I_levels = [-3, -1, +1, +3]
        Q_levels = [-7, -5, -3, -1, +1, +3, +5, +7]
    
    else if M == 128:
        // 128-QAM: 8x16 grid
        I_levels = [-7, -5, -3, -1, +1, +3, +5, +7]
        Q_levels = generate_range(-15, +15, step=2)
    
    else:
        error("Non-square constellation not defined for M = " + M)
    
    // Generate constellation
    constellation = empty_list()
    for I in I_levels:
        for Q in Q_levels:
            constellation.append(I + j*Q)
    
    constellation = array(constellation)
    
    // Normalize
    avg_power = mean(abs(constellation)^2)
    constellation = constellation / sqrt(avg_power)
    
    return constellation
end function
```
## Performance Comparison Function
```
function compare_qam_orders():
    // Compare different QAM orders
    
    M_values = [4, 16, 64, 256]
    
    print("QAM Order Comparison:")
    print("=" * 70)
    print(format_string("M", 8) + 
          format_string("Bits/Sym", 12) + 
          format_string("Norm Factor", 15) + 
          format_string("PAPR (dB)", 12) +
          format_string("Min Dist", 12))
    print("-" * 70)
    
    for M in M_values:
        constellation, norm = generate_qam_constellation(M)
        
        bits_per_sym = log2(M)
        avg_power = mean(abs(constellation)^2)
        peak_power = max(abs(constellation)^2)
        papr_db = 10 * log10(peak_power / avg_power)
        
        // Calculate minimum distance
        min_dist = calculate_minimum_distance(constellation)
        
        print(format_string(M, 8) + 
              format_string(bits_per_sym, 12) + 
              format_string(norm, 15, 4) + 
              format_string(papr_db, 12, 2) +
              format_string(min_dist, 12, 4))
    
    print("=" * 70)
end function

function calculate_minimum_distance(constellation):
    M = length(constellation)
    min_distance = infinity
    
    for i = 0 to M-2:
        for j = i+1 to M-1:
            distance = abs(constellation[i] - constellation[j])
            if distance > 1e-10 and distance < min_distance:
                min_distance = distance
    
    return min_distance
end function
```