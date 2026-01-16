# Signal Demodulation 
## Step 1: Low-Pass Filter Design
**Purpose:** Remove high-frequency mixing products after downconversion.
### Pseudocode for FIR Low-Pass Filter:
```
function design_lowpass_fir(cutoff_freq, sampling_rate, num_taps, window_type):
    // Design a FIR filter using windowing method
    // num_taps should be odd for symmetric filter (typically 51-101)
    
    // Normalize cutoff frequency
    normalized_cutoff = cutoff_freq / (sampling_rate / 2)
    
    // Create ideal sinc filter (infinite impulse response)
    filter_coeffs = zeros(num_taps)
    center = (num_taps - 1) / 2
    
    for n = 0 to num_taps-1:
        if n == center:
            filter_coeffs[n] = 2 * normalized_cutoff
        else:
            x = n - center
            // Sinc function
            filter_coeffs[n] = sin(2 * π * normalized_cutoff * x) / (π * x)
    end for
    
    // Apply window to reduce ripple
    window = create_window(num_taps, window_type)
    filter_coeffs = filter_coeffs * window
    
    // Normalize so DC gain = 1
    filter_coeffs = filter_coeffs / sum(filter_coeffs)
    
    return filter_coeffs
end function

function create_window(num_taps, window_type):
    // Common window functions
    window = zeros(num_taps)
    
    if window_type == "hamming":
        for n = 0 to num_taps-1:
            window[n] = 0.54 - 0.46 * cos(2 * π * n / (num_taps - 1))
    
    else if window_type == "hanning":
        for n = 0 to num_taps-1:
            window[n] = 0.5 * (1 - cos(2 * π * n / (num_taps - 1)))
    
    else if window_type == "blackman":
        for n = 0 to num_taps-1:
            window[n] = 0.42 - 0.5 * cos(2*π*n/(num_taps-1)) + 0.08 * cos(4*π*n/(num_taps-1))
    
    else:  // rectangular (no windowing)
        window = ones(num_taps)
    
    return window
end function

function apply_fir_filter(signal, filter_coeffs):
    // Convolve signal with filter coefficients
    num_samples = length(signal)
    num_taps = length(filter_coeffs)
    
    filtered = zeros(num_samples)
    
    // Direct convolution implementation
    for n = 0 to num_samples-1:
        filtered[n] = 0
        
        for k = 0 to num_taps-1:
            // Check bounds
            if (n - k) >= 0 and (n - k) < num_samples:
                filtered[n] = filtered[n] + signal[n - k] * filter_coeffs[k]
        end for
    end for
    
    return filtered
end function
```
### Alternative: IIR Butterworth Filter (mreo efficient)
```
function design_butterworth_lowpass(cutoff_freq, sampling_rate, order):
    // Design IIR Butterworth filter
    // Order typically 4-8 for good performance
    
    // Normalize cutoff frequency
    Wn = cutoff_freq / (sampling_rate / 2)
    
    // Calculate poles of Butterworth filter in s-domain
    poles_s = zeros(order) as complex
    for k = 0 to order-1:
        theta = π * (2*k + 1) / (2 * order) + π/2
        poles_s[k] = exp(j * theta)
    end for
    
    // Bilinear transform to z-domain
    poles_z = bilinear_transform(poles_s, Wn)
    
    // Convert poles/zeros to difference equation coefficients
    // a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... - a[1]*y[n-1] - a[2]*y[n-2] ...
    b_coeffs, a_coeffs = poles_to_coefficients(poles_z)
    
    return b_coeffs, a_coeffs
end function

function apply_iir_filter(signal, b_coeffs, a_coeffs):
    // Apply IIR filter using direct form II
    num_samples = length(signal)
    order = length(a_coeffs) - 1
    
    filtered = zeros(num_samples)
    state = zeros(order)  // Internal filter state
    
    for n = 0 to num_samples-1:
        // Feedforward (numerator)
        output = b_coeffs[0] * signal[n]
        for k = 1 to min(order, n):
            output = output + b_coeffs[k] * signal[n - k]
        
        // Feedback (denominator) - note a[0] is normalized to 1
        for k = 1 to order:
            if (n - k) >= 0:
                output = output - a_coeffs[k] * filtered[n - k]
        
        filtered[n] = output / a_coeffs[0]
    end for
    
    return filtered
end function
```
## Step 2: Matched Filter (for pulse-Shaped Signals)
**Purpose:** Maximise SNR and undo transmit pulse shaping.
### Pseudocode:
```
function create_matched_filter(pulse_shape_filter):
    // Matched filter is time-reversed conjugate of transmit filter
    matched_filter = conjugate(reverse(pulse_shape_filter))
    
    // Normalize for unity gain
    matched_filter = matched_filter / sqrt(sum(abs(matched_filter)^2))
    
    return matched_filter
end function

function root_raised_cosine_filter(beta, samples_per_symbol, num_symbols):
    // Root-raised cosine pulse shaping
    // beta = rolloff factor (0 to 1, typically 0.25-0.5)
    // num_symbols = filter span in symbols (typically 6-12)
    
    num_taps = num_symbols * samples_per_symbol
    if num_taps % 2 == 0:
        num_taps = num_taps + 1  // Make odd for symmetry
    
    filter = zeros(num_taps)
    center = (num_taps - 1) / 2
    
    for n = 0 to num_taps-1:
        t = (n - center) / samples_per_symbol
        
        // Handle special cases
        if t == 0:
            // t = 0
            filter[n] = 1 - beta + 4*beta/π
        
        else if abs(4*beta*t) == 1:
            // t = ±1/(4*beta)
            term1 = (1 + 2/π) * sin(π/(4*beta))
            term2 = (1 - 2/π) * cos(π/(4*beta))
            filter[n] = (beta/sqrt(2)) * (term1 + term2)
        
        else:
            // General case
            numerator = sin(π*t*(1-beta)) + 4*beta*t*cos(π*t*(1+beta))
            denominator = π*t*(1 - (4*beta*t)^2)
            filter[n] = numerator / denominator
    
    // Normalize for unit energy
    energy = sqrt(sum(filter^2))
    filter = filter / energy
    
    return filter
end function
```
## Step 3: Timing Recovery / Symbol Synchronisation
**Purpose:** Find optimal sampling points for symbol decisions.
### Pseudocode for Gardner Timing Error Detector (TED):
```
function gardner_timing_recovery(I_samples, Q_samples, samples_per_symbol):
    // Gardner algorithm for symbol timing recovery
    // Works well for many modulation types without needing decisions
    
    num_samples = length(I_samples)
    
    // Initialize
    mu = 0  // Fractional interval (0 to 1)
    mu_step = 1.0 / samples_per_symbol
    
    // Loop filter parameters for timing adjustment
    K1 = 0.01  // Proportional gain
    K2 = 0.0001  // Integral gain
    
    timing_error_filtered = 0
    
    symbol_indices = empty_list()
    timing_errors = empty_list()
    
    sample_index = samples_per_symbol  // Start after first symbol
    
    while sample_index < (num_samples - samples_per_symbol):
        // Get three samples: previous, current (mid), next
        // These correspond to: symbol n-1, mid-point, symbol n
        
        prev_index = round(sample_index - samples_per_symbol/2)
        curr_index = round(sample_index)
        next_index = round(sample_index + samples_per_symbol/2)
        
        if next_index >= num_samples:
            break
        
        // Get I and Q samples
        I_prev = I_samples[prev_index]
        I_curr = I_samples[curr_index]
        I_next = I_samples[next_index]
        
        Q_prev = Q_samples[prev_index]
        Q_curr = Q_samples[curr_index]
        Q_next = Q_samples[next_index]
        
        // Gardner timing error detector
        // Compares mid-point to average of neighbors
        timing_error_I = (I_next - I_prev) * I_curr
        timing_error_Q = (Q_next - Q_prev) * Q_curr
        timing_error = timing_error_I + timing_error_Q
        
        // Loop filter (PI controller)
        timing_error_filtered = timing_error_filtered + K2 * timing_error
        mu_adjustment = K1 * timing_error + timing_error_filtered
        
        // Update mu and advance to next symbol
        mu = mu + mu_step + mu_adjustment
        
        // Clamp mu to reasonable range
        if mu > 2.0:
            mu = 2.0
        if mu < -1.0:
            mu = -1.0
        
        sample_index = sample_index + samples_per_symbol + mu_adjustment * samples_per_symbol
        
        // Record symbol sampling point
        symbol_indices.append(curr_index)
        timing_errors.append(timing_error)
    end while
    
    return symbol_indices, timing_errors
end function

function simple_symbol_sampling(I_samples, Q_samples, samples_per_symbol, initial_offset):
    // Simplified approach: sample at fixed intervals
    // Assumes timing is already approximately correct
    
    num_samples = length(I_samples)
    symbol_indices = empty_list()
    
    // Start at initial offset (typically samples_per_symbol/2 after matched filter delay)
    index = initial_offset
    
    while index < num_samples:
        symbol_indices.append(round(index))
        index = index + samples_per_symbol
    end while
    
    return symbol_indices
end function
```
## Step 4: Carrier Frequency and Phase Recovery
**Purpose:** Correct for frequency offset and phase errors.
### Pseudocode for COstas Loop (for BPSK/QPSK):
```
function costas_loop_frequency_correction(I_samples, Q_samples, modulation_order):
    // Costas loop for carrier recovery
    // modulation_order: 2 for BPSK, 4 for QPSK
    
    num_samples = length(I_samples)
    
    // Initialize
    phase_estimate = 0
    frequency_estimate = 0
    
    // Loop filter gains
    alpha = 0.01  // Proportional gain (bandwidth)
    beta = 0.0001  // Integral gain (damping)
    
    // Corrected output
    I_corrected = zeros(num_samples)
    Q_corrected = zeros(num_samples)
    phase_estimates = zeros(num_samples)
    
    for n = 0 to num_samples-1:
        // Rotate by negative of phase estimate
        rotation = exp(-j * phase_estimate)
        complex_sample = (I_samples[n] + j * Q_samples[n]) * rotation
        
        I_corrected[n] = real(complex_sample)
        Q_corrected[n] = imag(complex_sample)
        
        // Phase error detector (depends on modulation)
        if modulation_order == 2:  // BPSK
            // Sign of I should be constant
            phase_error = sign(I_corrected[n]) * Q_corrected[n]
        
        else if modulation_order == 4:  // QPSK
            // Both I and Q should have consistent signs
            phase_error = sign(I_corrected[n]) * Q_corrected[n] - sign(Q_corrected[n]) * I_corrected[n]
        
        else:  // Generic (works for QAM too)
            // Decision-directed
            I_decided = decision(I_corrected[n])
            Q_decided = decision(Q_corrected[n])
            phase_error = I_corrected[n] * Q_decided - Q_corrected[n] * I_decided
        
        // Loop filter (PI controller)
        frequency_estimate = frequency_estimate + beta * phase_error
        phase_estimate = phase_estimate + frequency_estimate + alpha * phase_error
        
        // Wrap phase to [-π, π]
        phase_estimate = wrap_phase(phase_estimate)
        
        phase_estimates[n] = phase_estimate
    end for
    
    return I_corrected, Q_corrected, phase_estimates, frequency_estimate
end function

function wrap_phase(phase):
    // Wrap phase to [-π, π]
    while phase > π:
        phase = phase - 2*π
    while phase < -π:
        phase = phase + 2*π
    return phase
end function

function decision(value):
    // Simple slicer for constellation points
    if value >= 0:
        return 1
    else:
        return -1
end function
```
### Frequency Offset Estimation (Coarse):
```
function estimate_frequency_offset_fft(I_samples, Q_samples, samples_per_symbol):
    // Use FFT to estimate frequency offset
    // Works well for larger offsets
    
    complex_signal = I_samples + j * Q_samples
    
    // Raise to power of modulation order to remove modulation
    // For QPSK: raise to 4th power
    signal_power4 = complex_signal^4
    
    // Compute FFT
    fft_result = fft(signal_power4)
    magnitude = abs(fft_result)
    
    // Find peak
    peak_index = argmax(magnitude)
    
    num_samples = length(I_samples)
    
    // Convert bin to frequency
    if peak_index > num_samples/2:
        peak_index = peak_index - num_samples
    
    estimated_offset = peak_index / (4 * num_samples)  // Divide by 4 because we raised to 4th power
    
    return estimated_offset
end function
```
## Step 5: Symbol Detection / Decision
**Purpose:** Map received I/Q values to transmitted symbols.
### Pseudocode for BPSK Detector:
```
function detect_bpsk_symbols(I_symbols, Q_symbols):
    // BPSK uses only I channel
    num_symbols = length(I_symbols)
    detected_bits = zeros(num_symbols)
    
    for n = 0 to num_symbols-1:
        if I_symbols[n] >= 0:
            detected_bits[n] = 1
        else:
            detected_bits[n] = 0
    end for
    
    return detected_bits
end function
```
### Pseudocode for QPSK Detector:
```
function detect_qpsk_symbols(I_symbols, Q_symbols):
    // QPSK constellation:
    // 00 → (+1,+1), 01 → (-1,+1), 11 → (-1,-1), 10 → (+1,-1)
    
    num_symbols = length(I_symbols)
    detected_bits = zeros(num_symbols * 2)
    
    for n = 0 to num_symbols-1:
        I = I_symbols[n]
        Q = Q_symbols[n]
        
        // First bit from I
        if I >= 0:
            detected_bits[2*n] = 0
        else:
            detected_bits[2*n] = 1
        
        // Second bit from Q
        if Q >= 0:
            detected_bits[2*n + 1] = 0
        else:
            detected_bits[2*n + 1] = 1
    end for
    
    return detected_bits
end function
```
### Pseudocode for 16-QAM Detector:
```
function detect_16qam_symbols(I_symbols, Q_symbols):
    // 16-QAM: 4 levels per axis: -3, -1, +1, +3 (before normalization)
    
    num_symbols = length(I_symbols)
    detected_bits = zeros(num_symbols * 4)
    
    // Decision thresholds (assuming normalized constellation)
    norm_factor = sqrt(10)
    thresholds = [-2/norm_factor, 0, 2/norm_factor]
    
    for n = 0 to num_symbols-1:
        I = I_symbols[n]
        Q = Q_symbols[n]
        
        // Quantize I to nearest level
        I_level = quantize_to_level(I, thresholds)
        Q_level = quantize_to_level(Q, thresholds)
        
        // Convert levels (0,1,2,3) to 2-bit patterns
        I_bits = decimal_to_binary_2bit(I_level)
        Q_bits = decimal_to_binary_2bit(Q_level)
        
        // Pack into output
        detected_bits[4*n]     = I_bits[0]
        detected_bits[4*n + 1] = I_bits[1]
        detected_bits[4*n + 2] = Q_bits[0]
        detected_bits[4*n + 3] = Q_bits[1]
    end for
    
    return detected_bits
end function

function quantize_to_level(value, thresholds):
    // Quantize continuous value to discrete level
    // thresholds = [-2, 0, 2] gives 4 regions
    
    if value < thresholds[0]:
        return 0
    else if value < thresholds[1]:
        return 1
    else if value < thresholds[2]:
        return 2
    else:
        return 3
end function

function decimal_to_binary_2bit(decimal):
    // Convert 0,1,2,3 to 2-bit Gray code
    gray_mapping = [[0,0], [0,1], [1,1], [1,0]]
    return gray_mapping[decimal]
end function
```
### Pseudocode for General Minimum Distance Detector:
```
function detect_symbols_minimum_distance(I_symbols, Q_symbols, constellation):
    // constellation is array of complex constellation points
    // Works for any modulation scheme
    
    num_symbols = length(I_symbols)
    detected_indices = zeros(num_symbols)
    
    for n = 0 to num_symbols-1:
        received = I_symbols[n] + j * Q_symbols[n]
        
        // Find closest constellation point
        min_distance = infinity
        best_index = 0
        
        for k = 0 to length(constellation)-1:
            distance = abs(received - constellation[k])^2
            
            if distance < min_distance:
                min_distance = distance
                best_index = k
        end for
        
        detected_indices[n] = best_index
    end for
    
    // Convert indices to bits based on mapping
    detected_bits = indices_to_bits(detected_indices, constellation)
    
    return detected_bits
end function
```
## Complete Demolutation Chain Example
### Pseudocode:
```
function demodulate_signal(RF_signal, carrier_frequency, symbol_rate, sampling_rate, 
                           modulation_type, pulse_shape_filter):
    
    num_samples = length(RF_signal)
    time_axis = create_time_axis(num_samples, sampling_rate)
    samples_per_symbol = sampling_rate / symbol_rate
    
    // ============================================================
    // STEP 1: Downconvert from RF to baseband
    // ============================================================
    print("Step 1: Downconverting to baseband...")
    
    // Generate local oscillators (quadrature carriers)
    LO_I = cos(2 * π * carrier_frequency * time_axis)
    LO_Q = -sin(2 * π * carrier_frequency * time_axis)
    
    // Mix
    I_mixed = RF_signal * LO_I
    Q_mixed = RF_signal * LO_Q
    
    // ============================================================
    // STEP 2: Low-pass filtering
    // ============================================================
    print("Step 2: Low-pass filtering...")
    
    // Design filter
    cutoff_frequency = 1.5 * symbol_rate  // Bandwidth depends on pulse shaping
    num_taps = 101
    lpf_coeffs = design_lowpass_fir(cutoff_frequency, sampling_rate, num_taps, "hamming")
    
    // Apply filter
    I_baseband = apply_fir_filter(I_mixed, lpf_coeffs)
    Q_baseband = apply_fir_filter(Q_mixed, lpf_coeffs)
    
    // Account for filter delay
    filter_delay = (num_taps - 1) / 2
    
    // ============================================================
    // STEP 3: Matched filtering (if pulse shaping was used)
    // ============================================================
    if pulse_shape_filter is not null:
        print("Step 3: Matched filtering...")
        
        matched_filter = create_matched_filter(pulse_shape_filter)
        
        I_matched = apply_fir_filter(I_baseband, matched_filter)
        Q_matched = apply_fir_filter(Q_baseband, matched_filter)
        
        // Update delay
        matched_filter_delay = (length(matched_filter) - 1) / 2
        total_delay = filter_delay + matched_filter_delay
    else:
        I_matched = I_baseband
        Q_matched = Q_baseband
        total_delay = filter_delay
    
    // ============================================================
    // STEP 4: Carrier recovery (frequency and phase)
    // ============================================================
    print("Step 4: Carrier recovery...")
    
    // Coarse frequency offset estimation
    freq_offset = estimate_frequency_offset_fft(I_matched, Q_matched, samples_per_symbol)
    print("  Estimated frequency offset: " + freq_offset + " Hz")
    
    // Correct coarse frequency offset
    correction_phase = -2 * π * freq_offset * time_axis
    complex_corrected = (I_matched + j*Q_matched) * exp(j * correction_phase)
    I_freq_corrected = real(complex_corrected)
    Q_freq_corrected = imag(complex_corrected)
    
    // Fine phase tracking with Costas loop
    modulation_order = get_modulation_order(modulation_type)
    I_corrected, Q_corrected, phase_track, residual_freq = 
        costas_loop_frequency_correction(I_freq_corrected, Q_freq_corrected, modulation_order)
    
    print("  Residual frequency error: " + residual_freq + " Hz")
    
    // ============================================================
    // STEP 5: Timing recovery
    // ============================================================
    print("Step 5: Symbol timing recovery...")
    
    // Account for filter delays and find first symbol
    initial_offset = round(total_delay + samples_per_symbol/2)
    
    // Use Gardner TED for timing recovery
    symbol_indices, timing_errors = gardner_timing_recovery(I_corrected, Q_corrected, 
                                                              samples_per_symbol)
    
    // OR use simple sampling if timing is good
    // symbol_indices = simple_symbol_sampling(I_corrected, Q_corrected, 
    //                                          samples_per_symbol, initial_offset)
    
    print("  Recovered " + length(symbol_indices) + " symbols")
    
    // ============================================================
    // STEP 6: Sample at symbol times
    // ============================================================
    print("Step 6: Extracting symbol samples...")
    
    num_symbols = length(symbol_indices)
    I_symbols = zeros(num_symbols)
    Q_symbols = zeros(num_symbols)
    
    for n = 0 to num_symbols-1:
        idx = symbol_indices[n]
        if idx < num_samples:
            I_symbols[n] = I_corrected[idx]
            Q_symbols[n] = Q_corrected[idx]
    end for
    
    // ============================================================
    // STEP 7: Symbol detection
    // ============================================================
    print("Step 7: Symbol detection...")
    
    if modulation_type == "BPSK":
        detected_bits = detect_bpsk_symbols(I_symbols, Q_symbols)
    
    else if modulation_type == "QPSK":
        detected_bits = detect_qpsk_symbols(I_symbols, Q_symbols)
    
    else if modulation_type == "16QAM":
        detected_bits = detect_16qam_symbols(I_symbols, Q_symbols)
    
    else:
        error("Unsupported modulation type")
    
    print("Demodulation complete!")
    print("  Total bits recovered: " + length(detected_bits))
    
    // Return results
    return {
        detected_bits: detected_bits,
        I_symbols: I_symbols,
        Q_symbols: Q_symbols,
        I_baseband: I_corrected,
        Q_baseband: Q_corrected,
        symbol_indices: symbol_indices,
        timing_errors: timing_errors,
        phase_track: phase_track,
        frequency_offset: freq_offset + residual_freq
    }
end function

function get_modulation_order(modulation_type):
    if modulation_type == "BPSK":
        return 2
    else if modulation_type == "QPSK":
        return 4
    else if modulation_type == "8PSK":
        return 8
    else if modulation_type == "16QAM":
        return 4  // For Costas loop purposes
    else:
        return 4  // Default
end function
```
## Performace Metrics
### Calculate Bit Error Rate (BER):
```
function calculate_ber(transmitted_bits, received_bits):
    // Align sequences if needed
    delay = find_alignment_delay(transmitted_bits, received_bits)
    
    // Count bit errors
    num_errors = 0
    num_compared = 0
    
    for n = delay to min(length(transmitted_bits), length(received_bits))-1:
        if transmitted_bits[n] != received_bits[n - delay]:
            num_errors = num_errors + 1
        num_compared = num_compared + 1
    end for
    
    BER = num_errors / num_compared
    
    return BER, num_errors, num_compared
end function

function find_alignment_delay(reference, received):
    // Cross-correlate to find delay
    max_search = 100  // symbols to search
    best_delay = 0
    best_correlation = 0
    
    for delay = 0 to max_search:
        correlation = 0
        for n = delay to min(length(reference), length(received))-1:
            if reference[n] == received[n - delay]:
                correlation = correlation + 1
        
        if correlation > best_correlation:
            best_correlation = correlation
            best_delay = delay
    
    return best_delay
end function
```
### Calculate Error Vector Magnitude (EVM):
```
function calculate_evm(I_received, Q_received, constellation):
    // Measure how far received symbols are from ideal
    
    num_symbols = length(I_received)
    total_error_power = 0
    total_reference_power = 0
    
    for n = 0 to num_symbols-1:
        received = I_received[n] + j * Q_received[n]
        
        // Find nearest constellation point
        ideal = find_nearest_constellation_point(received, constellation)
        
        // Error vector
        error = received - ideal
        error_power = abs(error)^2
        reference_power = abs(ideal)^2
        
        total_error_power = total_error_power + error_power
        total_reference_power = total_reference_power + reference_power
    end for
    
    EVM_rms = sqrt(total_error_power / num_symbols)
    EVM_percent = 100 * sqrt(total_error_power / total_reference_power)
    
    return EVM_rms, EVM_percent
end function
```