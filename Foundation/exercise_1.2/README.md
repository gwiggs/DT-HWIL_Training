# Understanding Complex Exponentials
A complex exponential signal has the form: x(t) = A · e^(j·2π·f·t)
where:

- A = amplitude (can be complex for phase offset)
- j = √(-1) (imaginary unit)
- f = frequency in Hz
- t = time

Euler's formula tells us this equals: e^(j·θ) = cos(θ) + j·sin(θ)
So your signal has:
- Real part: A·cos(2πft)
- Imaginary part: A·sin(2πft)

This is why a complex exponential is sometimes called a "rotating phasor" - it traces a circle in the complex plane.

## Basic Complex Exponential Generation
Pseudocode:
```
// Define signal parameters
sampling_rate = 44100          // Hz
signal_frequency = 1000        // Hz
duration = 1.0                 // seconds
amplitude = 1.0                // can be complex: amplitude * e^(j*initial_phase)

// Create time axis
num_samples = sampling_rate * duration
time_axis = linspace(0, duration, num_samples)
// Each sample at time_axis[n] = n / sampling_rate

// Generate complex exponential
// Method 1: Using Euler's formula explicitly
omega = 2 * π * signal_frequency
complex_signal = zeros(num_samples) as complex array

for n = 0 to num_samples-1:
    t = time_axis[n]
    theta = omega * t
    
    // Real and imaginary parts
    real_part = amplitude * cos(theta)
    imag_part = amplitude * sin(theta)
    
    complex_signal[n] = real_part + j * imag_part
end for
```
## Adding Initial Phase Offset
```
// To start at a specific phase
initial_phase = π/4  // radians (45 degrees)

omega = 2 * π * signal_frequency

for n = 0 to num_samples-1:
    t = time_axis[n]
    theta = omega * t + initial_phase
    
    complex_signal[n] = amplitude * exp(j * theta)
    // Or: amplitude * (cos(theta) + j*sin(theta))
end for
```
## Extracting Compenents
```
// Separate real and imaginary parts
real_component = real(complex_signal)      // cos part
imaginary_component = imag(complex_signal) // sin part

// Compute magnitude and phase at each point
magnitude = abs(complex_signal)            // sqrt(real² + imag²)
phase_angle = angle(complex_signal)        // atan2(imag, real)
```

# In-phase/Quadrature Representation
## Core Concept:
I/Q representation expresses any bandpass signal as two baseband components:
- I (In-phase): Real part, aligned with carrier
- Q (Quadrature): Imaginary part, 90° phase-shifted from carrier

**Key insight**: Any modulated RF signal can be written as:
```
s(t) = I(t)·cos(2πf_c·t) - Q(t)·sin(2πf_c·t)
```
where:
- f_c = carrier frequency
- I(t) = in-phase baseband signal
- Q(t) = quadrature baseband signal

**Complex baseband represenation:**
```
s_baseband(t) = I(t) + j·Q(t)
s_passband(t) = Real{s_baseband(t) · e^(j·2π·f_c·t)}
```
## Why I/Q Representation?

1. **Separates carrier from information:** Information is in I(t) and Q(t)
2. **Enables digital processing:** Work at baseband instead of RF frequencies
3. **Efficient:** Two real signals instead of one high-frequency signal
4. **Flexible:** Supports arbitrary modulation schemes
5. **Practical:** How SDRs and modern radios actually work
## Generating I/Q Signals
### General Framework
Pseudocode:
```
// Parameters
carrier_frequency = 10000      // Hz (RF carrier)
sampling_rate = 100000         // Hz (must be >> carrier_frequency)
symbol_rate = 1000             // Symbols per second
duration = 1.0                 // seconds

// Create time axis
num_samples = sampling_rate * duration
time_axis = linspace(0, duration, num_samples)

// Step 1: Generate baseband I and Q components
// (depends on modulation scheme - see examples below)
I_baseband = generate_I_component()  // Real-valued
Q_baseband = generate_Q_component()  // Real-valued

// Step 2: Create complex baseband signal
complex_baseband = I_baseband + j * Q_baseband

// Step 3: Upconvert to passband (if needed for transmission)
carrier = exp(j * 2 * π * carrier_frequency * time_axis)
complex_passband = complex_baseband * carrier

// Step 4: Get real RF signal for transmission
RF_signal = real(complex_passband)
// Equivalent to: I(t)·cos(ωt) - Q(t)·sin(ωt)
```
### Example 1: BPSK (Binary Phase Shift Keying)
**Concept:** Information encoded in phase (0° or 180°)
Pseudocode:
```
// Generate random binary data
num_symbols = symbol_rate * duration
binary_data = random_binary(num_symbols)  // Array of 0s and 1s

// BPSK mapping: 0 → -1, 1 → +1
symbols = zeros(num_symbols)
for n = 0 to num_symbols-1:
    if binary_data[n] == 0:
        symbols[n] = -1
    else:
        symbols[n] = +1
end for

// Upsample symbols to match sampling rate
samples_per_symbol = sampling_rate / symbol_rate

I_baseband = zeros(num_samples)
Q_baseband = zeros(num_samples)  // BPSK uses only I channel

for n = 0 to num_symbols-1:
    start_index = n * samples_per_symbol
    end_index = (n + 1) * samples_per_symbol
    
    // Fill samples for this symbol duration
    for k = start_index to end_index-1:
        I_baseband[k] = symbols[n]
        Q_baseband[k] = 0  // No quadrature component
    end for
end for

// Result: I toggles between ±1, Q stays at 0
```
### Example 2: QPSK (Quadrature Phase Shift Keying)
**Concept:** Information encoded in four phase states (0°, 90°, 180°, 270°). Uses both I and Q channels → twice the data rate of BPSK.
Pseudocode:
```
// Generate random binary data (2 bits per symbol)
num_symbols = symbol_rate * duration
binary_data = random_binary(num_symbols * 2)

// QPSK constellation mapping
// 00 → (I=+1, Q=+1) → 45°
// 01 → (I=-1, Q=+1) → 135°
// 11 → (I=-1, Q=-1) → 225°
// 10 → (I=+1, Q=-1) → 315°

symbols_I = zeros(num_symbols)
symbols_Q = zeros(num_symbols)

for n = 0 to num_symbols-1:
    // Take two bits per symbol
    bit1 = binary_data[2*n]
    bit2 = binary_data[2*n + 1]
    
    // Map to constellation
    if bit1 == 0 and bit2 == 0:
        symbols_I[n] = +1/sqrt(2)
        symbols_Q[n] = +1/sqrt(2)
    else if bit1 == 0 and bit2 == 1:
        symbols_I[n] = -1/sqrt(2)
        symbols_Q[n] = +1/sqrt(2)
    else if bit1 == 1 and bit2 == 1:
        symbols_I[n] = -1/sqrt(2)
        symbols_Q[n] = -1/sqrt(2)
    else:  // bit1 == 1 and bit2 == 0
        symbols_I[n] = +1/sqrt(2)
        symbols_Q[n] = -1/sqrt(2)
end for

// Upsample to sampling rate
samples_per_symbol = sampling_rate / symbol_rate

I_baseband = upsample_and_hold(symbols_I, samples_per_symbol)
Q_baseband = upsample_and_hold(symbols_Q, samples_per_symbol)

// Helper function
function upsample_and_hold(symbols, samples_per_symbol):
    output = zeros(length(symbols) * samples_per_symbol)
    for n = 0 to length(symbols)-1:
        for k = 0 to samples_per_symbol-1:
            output[n * samples_per_symbol + k] = symbols[n]
    return output
```
### Example 3: QAM (Quadrature Amplitude Modulation)
**Concept:** Both amplitude AND phase carry information. Common: 16-QAM, 64-QAM, 256-QAM
Pseudocode for 16-QAM:
```
// 16-QAM: 4 bits per symbol, 16 constellation points
// Points arranged in a 4x4 grid

num_symbols = symbol_rate * duration
binary_data = random_binary(num_symbols * 4)  // 4 bits per symbol

// QAM levels (normalized)
levels = [-3, -1, +1, +3]

symbols_I = zeros(num_symbols)
symbols_Q = zeros(num_symbols)

for n = 0 to num_symbols-1:
    // Extract 4 bits for this symbol
    bits = binary_data[4*n : 4*n+4]
    
    // First 2 bits determine I level
    I_bits = bits[0:2]
    I_index = binary_to_decimal(I_bits)
    symbols_I[n] = levels[I_index]
    
    // Last 2 bits determine Q level
    Q_bits = bits[2:4]
    Q_index = binary_to_decimal(Q_bits)
    symbols_Q[n] = levels[Q_index]
end for

// Normalize for constant average power
normalization = sqrt(10)  // For 16-QAM
symbols_I = symbols_I / normalization
symbols_Q = symbols_Q / normalization

// Upsample
I_baseband = upsample_and_hold(symbols_I, samples_per_symbol)
Q_baseband = upsample_and_hold(symbols_Q, samples_per_symbol)
```
### Example 4: Frequency Shift Keying (FSK) in I/Q
**Concept:** Different frequencies for different symbols
Pseudocode for Binary FSK:
```
// Two frequencies for binary FSK
freq_0 = 1000  // Hz (represents bit 0)
freq_1 = 2000  // Hz (represents bit 1)

num_symbols = symbol_rate * duration
binary_data = random_binary(num_symbols)

complex_baseband = zeros(num_samples) as complex

sample_index = 0
for n = 0 to num_symbols-1:
    // Choose frequency based on bit
    if binary_data[n] == 0:
        freq = freq_0
    else:
        freq = freq_1
    
    // Generate complex exponential for symbol duration
    for k = 0 to samples_per_symbol-1:
        t = sample_index / sampling_rate
        phase = 2 * π * freq * t
        complex_baseband[sample_index] = exp(j * phase)
        sample_index = sample_index + 1
end for
end for

I_baseband = real(complex_baseband)
Q_baseband = imag(complex_baseband)
```
## Pulse Shaping
(Important for Realistic Signals)
**Problem:** Rectangular symbols create wide spectral spread
**Solution:** Filter symbols with raised cosine or root-raised cosine
Pseudocode:
```
// Create pulse shaping filter
function root_raised_cosine_filter(beta, symbol_duration, num_taps):
    // beta = rolloff factor (0 to 1)
    // num_taps = filter length (typically 8-12 symbols)
    
    filter = zeros(num_taps * samples_per_symbol)
    center = length(filter) / 2
    
    for n = 0 to length(filter)-1:
        t = (n - center) / samples_per_symbol
        
        if t == 0:
            filter[n] = (1 + beta*(4/π - 1))
        else if abs(t) == 1/(4*beta):
            filter[n] = (beta/sqrt(2)) * ((1+2/π)*sin(π/(4*beta)) + (1-2/π)*cos(π/(4*beta)))
        else:
            numerator = sin(π*t*(1-beta)) + 4*beta*t*cos(π*t*(1+beta))
            denominator = π*t*(1-(4*beta*t)^2)
            filter[n] = numerator / denominator
    
    // Normalize
    filter = filter / sqrt(sum(filter^2))
    return filter
end function

// Apply pulse shaping
rrc_filter = root_raised_cosine_filter(beta=0.35, symbol_duration, num_taps=10)

// Convolve I and Q with filter
I_shaped = convolve(I_baseband, rrc_filter)
Q_shaped = convolve(Q_baseband, rrc_filter)

// Note: Convolution adds delay and extends signal length
```
# Visualisation of I/Q Signals
## 1. Constellation Diagram
```
// Shows symbol mapping in I-Q plane
create_plot()
scatter(symbols_I, symbols_Q)
xlabel("I (In-phase)")
ylabel("Q (Quadrature)")
title("Constellation Diagram")
axis_equal()
grid_on()
add_reference_circle(radius=1)  // Unit circle reference
```
## 2. Time Domain I and Q
```
create_subplot(2, 1, 1)
plot(time_axis, I_baseband)
ylabel("I(t)")
title("I/Q Baseband Components")
grid_on()

create_subplot(2, 1, 2)
plot(time_axis, Q_baseband)
xlabel("Time (s)")
ylabel("Q(t)")
grid_on()
```
## 3. Eye Diagram (shows ISI and timing)
```
// Overlay multiple symbol periods
samples_per_symbol = sampling_rate / symbol_rate
num_traces = 100

create_plot()
for n = 0 to num_traces-1:
    start = n * samples_per_symbol
    end = start + 2 * samples_per_symbol  // Two symbols
    
    if end <= num_samples:
        time_segment = linspace(0, 2, 2*samples_per_symbol)
        plot(time_segment, I_baseband[start:end], alpha=0.3)
end for
xlabel("Normalized Time (symbols)")
ylabel("Amplitude")
title("Eye Diagram - I Channel")
```
## 4. Spectrogram (time-frequency)
```
// Shows how spectrum changes over time
spectrogram(I_baseband + j*Q_baseband, 
            window_size=256,
            overlap=128,
            sampling_rate)
xlabel("Time (s)")
ylabel("Frequency (Hz)")
title("I/Q Spectrogram")

```
## Demodulation: Extracting I/Q from RF
Pseudocode for receiver:
```
// Received RF signal
RF_signal = received_waveform

// Step 1: Generate local oscillator (same frequency as transmitter)
LO_I = cos(2 * π * carrier_frequency * time_axis)
LO_Q = -sin(2 * π * carrier_frequency * time_axis)

// Step 2: Mix down to baseband
I_mixed = RF_signal * LO_I
Q_mixed = RF_signal * LO_Q

// Step 3: Low-pass filter to remove high-frequency components
cutoff_frequency = 2 * symbol_rate  // Or slightly higher
I_baseband_recovered = lowpass_filter(I_mixed, cutoff_frequency)
Q_baseband_recovered = lowpass_filter(Q_mixed, cutoff_frequency)

// Step 4: Matched filter (if pulse shaping was used)
I_filtered = convolve(I_baseband_recovered, matched_filter)
Q_filtered = convolve(Q_baseband_recovered, matched_filter)

// Step 5: Sample at symbol rate
samples_per_symbol = sampling_rate / symbol_rate
I_symbols = I_filtered[::samples_per_symbol]  // Downsample
Q_symbols = Q_filtered[::samples_per_symbol]

// Step 6: Decision/detection based on constellation
decoded_bits = detect_symbols(I_symbols, Q_symbols)
```
# Key Properties to Verify
After generating I/Q signals:
1. Constellation points: Should match expected modulation
2. Average power: mean(I²) + mean(Q²) should be consistent
3. Spectral occupancy: Bandwidth should match symbol rate × (1 + rolloff)
4. Peak-to-Average Power Ratio (PAPR): Important for amplifier design
5. Eye opening: Clear eye means good signal quality

## Advanced Considerations
### Carrier offset/frequency error:
```
// Simulating frequency offset
frequency_offset = 50  // Hz
phase_error = 2 * π * frequency_offset * time_axis

complex_baseband_with_offset = complex_baseband * exp(j * phase_error)
```
### Phase noise:
```
// Add random phase variations
phase_noise_std = 0.1  // radians
phase_noise = gaussian_noise(num_samples, mean=0, std=phase_noise_std)

complex_baseband_noisy = complex_baseband * exp(j * cumsum(phase_noise))
```
### AWGN channel:
```
// Add white Gaussian noise
SNR_dB = 10  // Signal-to-noise ratio in dB
signal_power = mean(abs(complex_baseband)^2)
noise_power = signal_power / (10^(SNR_dB/10))

noise_I = gaussian_noise(num_samples, mean=0, std=sqrt(noise_power/2))
noise_Q = gaussian_noise(num_samples, mean=0, std=sqrt(noise_power/2))

received_signal = complex_baseband + (noise_I + j*noise_Q)
```