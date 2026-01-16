# Visualisation Priority by Modulation Type
## BPSK
**Most Important (in order):**
1. Eye Diagram - I Channel
- Why: BPSK only uses I channel, so this is critical
- What to look for:
  - Wide eye opening → good timing
  - Two distinct levels (+1, -1)
  - Crossing point at 50% amplitude
- When to use: Diagnosing timing issues, ISI, noise
2. Constellation Diagram
- Why: Should show two clear points on real axis
- What to look for:
 - Points at (+1, 0) and (-1, 0)
 - Tight clustering → low noise
 - Spread along I axis → good
 - Spread in Q direction → carrier phase error
- When to use: Checking SNR, phase lock quality
3. Phase Trajectory Over Time
- Why: Shows phase transitions (0° ↔ 180°)
- What to look for:
 - Square wave between 0° and 180°
 - Smooth transitions if pulse-shaped
- When to use: Verifying modulation, checking phase continuity
### Pseudocode for BPSK-specific visualisations:
```
// Eye diagram - I channel only
function plot_bpsk_eye_diagram(I_samples, samples_per_symbol):
    create_figure()
    
    // Overlay 2 symbol periods
    trace_length = 2 * samples_per_symbol
    num_traces = floor(length(I_samples) / samples_per_symbol) - 2
    
    for n = 0 to num_traces-1:
        start = n * samples_per_symbol
        end = start + trace_length
        
        time_normalized = linspace(0, 2, trace_length)
        plot(time_normalized, I_samples[start:end], alpha=0.3, color='blue')
    end for
    
    // Add decision threshold line
    plot_horizontal_line(y=0, color='red', linestyle='dashed')
    
    xlabel("Time (symbols)")
    ylabel("I Amplitude")
    title("BPSK Eye Diagram - I Channel")
    grid_on()
end function

// Phase trajectory
function plot_bpsk_phase_trajectory(I_samples, Q_samples, time_axis):
    // Calculate instantaneous phase
    phase = atan2(Q_samples, I_samples)
    
    // Unwrap phase to avoid discontinuities
    phase_unwrapped = unwrap(phase)
    
    create_figure()
    plot(time_axis, phase_unwrapped * 180/π)  // Convert to degrees
    ylabel("Phase (degrees)")
    xlabel("Time (s)")
    title("BPSK Phase Trajectory")
    
    // Mark ideal phase values
    plot_horizontal_line(y=0, color='red', linestyle='dashed', label='0°')
    plot_horizontal_line(y=180, color='red', linestyle='dashed', label='180°')
    grid_on()
end function
```
## QPSK
**Most Important:**
1. Constellation Diagram
- Why: QPSK's four states are best seen here
- What to look for:
 - Four tight clusters at 45°, 135°, 225°, 315°
 - Equal magnitude for all points
 - Rotation → carrier frequency offset
 - Elongation along axis → phase noise
- When to use: Primary diagnostic tool for QPSK
2. Eye Diagram - Both I and Q
- Why: Both channels carry information equally
- What to look for:
 - Three distinct levels in each eye (transitions between ±1)
 - Equal eye opening in I and Q
 - Synchronized zero crossings
- When to use: Timing recovery verification
3. Trajectory Diagram (I/Q over time)
- Why: Shows constellation transitions
- What to look for:
 - Paths between constellation points
 - Smooth transitions if pulse-shaped
 - No paths through origin (good phase shaping)
- When to use: Checking pulse shaping effectiveness
### Pseudocode for QPSK visualizations:
```
function plot_qpsk_constellation(I_symbols, Q_symbols):
    create_figure()
    
    // Scatter plot of symbol decisions
    scatter(I_symbols, Q_symbols, alpha=0.5, s=20)
    
    // Overlay ideal constellation points
    ideal_I = [1, -1, -1, 1] / sqrt(2)
    ideal_Q = [1, 1, -1, -1] / sqrt(2)
    scatter(ideal_I, ideal_Q, color='red', marker='x', s=200, linewidth=3)
    
    // Draw unit circle for reference
    theta = linspace(0, 2*π, 100)
    plot(cos(theta)/sqrt(2), sin(theta)/sqrt(2), 'k--', alpha=0.3)
    
    xlabel("I (In-phase)")
    ylabel("Q (Quadrature)")
    title("QPSK Constellation Diagram")
    axis_equal()
    grid_on()
    
    // Add decision boundaries
    plot_vertical_line(x=0, color='gray', linestyle=':', alpha=0.5)
    plot_horizontal_line(y=0, color='gray', linestyle=':', alpha=0.5)
end function

function plot_qpsk_dual_eye_diagram(I_samples, Q_samples, samples_per_symbol):
    create_figure_with_subplots(2, 1)
    
    trace_length = 2 * samples_per_symbol
    num_traces = floor(length(I_samples) / samples_per_symbol) - 2
    
    // I channel eye
    select_subplot(1)
    for n = 0 to num_traces-1:
        start = n * samples_per_symbol
        end = start + trace_length
        time_normalized = linspace(0, 2, trace_length)
        plot(time_normalized, I_samples[start:end], alpha=0.3, color='blue')
    end for
    ylabel("I Amplitude")
    title("QPSK Eye Diagram - I Channel")
    grid_on()
    
    // Q channel eye
    select_subplot(2)
    for n = 0 to num_traces-1:
        start = n * samples_per_symbol
        end = start + trace_length
        time_normalized = linspace(0, 2, trace_length)
        plot(time_normalized, Q_samples[start:end], alpha=0.3, color='red')
    end for
    xlabel("Time (symbols)")
    ylabel("Q Amplitude")
    title("QPSK Eye Diagram - Q Channel")
    grid_on()
end function

function plot_iq_trajectory(I_samples, Q_samples, num_symbols_to_plot):
    // Plot signal path through constellation over time
    create_figure()
    
    // Limit to reasonable number of points
    end_sample = min(num_symbols_to_plot * samples_per_symbol, length(I_samples))
    
    // Plot trajectory
    plot(I_samples[0:end_sample], Q_samples[0:end_sample], 
         linewidth=0.5, alpha=0.6)
    
    // Mark start and end
    scatter(I_samples[0], Q_samples[0], color='green', s=100, label='Start', zorder=5)
    scatter(I_samples[end_sample-1], Q_samples[end_sample-1], 
            color='red', s=100, label='End', zorder=5)
    
    // Ideal constellation
    ideal_I = [1, -1, -1, 1] / sqrt(2)
    ideal_Q = [1, 1, -1, -1] / sqrt(2)
    scatter(ideal_I, ideal_Q, color='black', marker='x', s=200, linewidth=3)
    
    xlabel("I (In-phase)")
    ylabel("Q (Quadrature)")
    title("QPSK I/Q Trajectory")
    axis_equal()
    grid_on()
    legend()
end function
```
## 16-QAM and Higher-Order QAM
**Most Important:**
1. Constellation Diagram with Decision Boundaries
- Why: Many constellation points need clear visualization
- What to look for:
 - 16 distinct clusters in grid pattern
 - Equal spacing between points
 - Clusters don't overlap
 - Outer points have higher error rates
- When to use: Primary diagnostic, especially for EVM
2. EVM vs Symbol Index
- Why: Shows how error varies over time
- What to look for:
 - Consistent low EVM → good link
 - Periodic variations → synchronization issues
 - Gradual drift → frequency offset
- When to use: Performance monitoring, link quality
3. Error Vector Magnitude Histogram
- Why: Statistical distribution of errors
- What to look for:
 - Peak near zero
 - Narrow distribution → low noise
 - Long tail → occasional deep fades
- When to use: Link budget analysis
### Pseudocode for QAM visualizations:
```
function plot_qam_constellation_with_boundaries(I_symbols, Q_symbols, modulation_order):
    create_figure()
    
    // Generate ideal constellation
    constellation = generate_qam_constellation(modulation_order)
    
    // Plot received symbols
    scatter(I_symbols, Q_symbols, alpha=0.3, s=10, color='blue')
    
    // Plot ideal points
    scatter(real(constellation), imag(constellation), 
            color='red', marker='x', s=200, linewidth=3, label='Ideal')
    
    // Draw decision boundaries
    if modulation_order == 16:
        levels = [-2, 0, 2]  // Normalized
        
        for threshold in levels:
            plot_vertical_line(x=threshold, color='gray', 
                             linestyle='--', alpha=0.5)
            plot_horizontal_line(y=threshold, color='gray', 
                               linestyle='--', alpha=0.5)
    
    xlabel("I (In-phase)")
    ylabel("Q (Quadrature)")
    title(str(modulation_order) + "-QAM Constellation")
    axis_equal()
    grid_on()
    legend()
    
    // Zoom to appropriate range
    margin = 1.2
    max_val = max(abs(constellation))
    set_xlim(-max_val*margin, max_val*margin)
    set_ylim(-max_val*margin, max_val*margin)
end function

function plot_evm_vs_time(I_symbols, Q_symbols, constellation):
    num_symbols = length(I_symbols)
    
    evm_values = zeros(num_symbols)
    
    // Calculate EVM for each symbol
    for n = 0 to num_symbols-1:
        received = I_symbols[n] + j * Q_symbols[n]
        ideal = find_nearest_constellation_point(received, constellation)
        
        error_magnitude = abs(received - ideal)
        reference_magnitude = abs(ideal)
        
        evm_values[n] = 100 * error_magnitude / reference_magnitude  // Percentage
    end for
    
    create_figure()
    plot(evm_values)
    xlabel("Symbol Index")
    ylabel("EVM (%)")
    title("Error Vector Magnitude vs Time")
    grid_on()
    
    // Add mean line
    mean_evm = mean(evm_values)
    plot_horizontal_line(y=mean_evm, color='red', linestyle='--', 
                        label='Mean EVM = ' + format(mean_evm, '.2f') + '%')
    legend()
end function

function plot_evm_histogram(I_symbols, Q_symbols, constellation):
    // Calculate all EVM values
    evm_values = calculate_all_evm(I_symbols, Q_symbols, constellation)
    
    create_figure()
    histogram(evm_values, bins=50, density=True, alpha=0.7)
    
    xlabel("EVM (%)")
    ylabel("Probability Density")
    title("EVM Distribution")
    grid_on()
    
    // Add statistics
    mean_evm = mean(evm_values)
    std_evm = std(evm_values)
    
    text_x = max(evm_values) * 0.6
    text_y = get_y_max() * 0.8
    
    add_text(text_x, text_y, 
            "Mean: " + format(mean_evm, '.2f') + "%\n" +
            "Std: " + format(std_evm, '.2f') + "%")
end function
```
## FSK
**Most Important:**

1. Spectrogram (Time-Frequency)
- Why: Shows frequency changes over time directly
- What to look for:
 - Discrete frequency bands
 - Sharp transitions between frequencies
 - No frequency drift within symbols
- When to use: Primary diagnostic for FSK
2. Instantaneous Frequency vs Time
- Why: Direct visualization of frequency modulation
- What to look for:
 - Square wave pattern
 - Stable frequency levels
 - Clean transitions
- When to use: Verifying frequency accuracy
3. FFT (averaged over symbol period)
- Why: Shows spectral separation of tones
- What to look for:
 - Distinct peaks at each FSK frequency
 - No spectral overlap
 - Adequate frequency separation
- When to use: Checking frequency spacing
### Pseudocode for FSK visualizations:
```
function plot_fsk_spectrogram(I_samples, Q_samples, sampling_rate):
    complex_signal = I_samples + j * Q_samples
    
    // Spectrogram parameters
    window_size = 256
    overlap = window_size * 3/4
    
    create_figure()
    spectrogram(complex_signal, 
                fs=sampling_rate,
                window_size=window_size,
                overlap=overlap,
                colormap='viridis')
    
    xlabel("Time (s)")
    ylabel("Frequency (Hz)")
    title("FSK Spectrogram")
    colorbar(label="Power (dB)")
end function

function plot_instantaneous_frequency(I_samples, Q_samples, sampling_rate):
    // Calculate instantaneous frequency from phase derivative
    complex_signal = I_samples + j * Q_samples
    
    // Instantaneous phase
    phase = atan2(Q_samples, I_samples)
    phase_unwrapped = unwrap(phase)
    
    // Frequency is derivative of phase / (2π)
    inst_frequency = zeros(length(phase_unwrapped) - 1)
    
    for n = 1 to length(phase_unwrapped)-1:
        phase_diff = phase_unwrapped[n] - phase_unwrapped[n-1]
        inst_frequency[n-1] = (phase_diff * sampling_rate) / (2 * π)
    end for
    
    time_axis = linspace(0, length(inst_frequency)/sampling_rate, 
                         length(inst_frequency))
    
    create_figure()
    plot(time_axis, inst_frequency)
    xlabel("Time (s)")
    ylabel("Instantaneous Frequency (Hz)")
    title("FSK Instantaneous Frequency")
    grid_on()
    
    // Mark expected frequencies if known
    // plot_horizontal_line(y=freq_0, color='red', linestyle='--', label='f0')
    // plot_horizontal_line(y=freq_1, color='red', linestyle='--', label='f1')
end function

function plot_fsk_spectrum_waterfall(I_samples, Q_samples, sampling_rate, 
                                     samples_per_symbol):
    // Show FFT evolving over time (waterfall plot)
    complex_signal = I_samples + j * Q_samples
    
    fft_size = 512
    num_ffts = floor(length(complex_signal) / samples_per_symbol)
    
    waterfall_data = zeros(num_ffts, fft_size)
    
    for n = 0 to num_ffts-1:
        start = n * samples_per_symbol
        end = start + fft_size
        
        if end <= length(complex_signal):
            segment = complex_signal[start:end]
            fft_result = fft(segment, fft_size)
            waterfall_data[n, :] = 20 * log10(abs(fft_result) + 1e-10)
    end for
    
    // Create frequency axis
    freq_axis = linspace(-sampling_rate/2, sampling_rate/2, fft_size)
    time_axis = linspace(0, num_ffts * samples_per_symbol / sampling_rate, num_ffts)
    
    create_figure()
    imshow(waterfall_data, 
           extent=[freq_axis[0], freq_axis[-1], time_axis[-1], time_axis[0]],
           aspect='auto',
           colormap='jet')
    
    xlabel("Frequency (Hz)")
    ylabel("Time (s)")
    title("FSK Waterfall Plot")
    colorbar(label="Magnitude (dB)")
end function
```
## Universal "Debug Dashboard" for Any Modulation
**For comprehensive analysis, create a single figure with multiple subplots:**
```
function create_comprehensive_dashboard(I_samples, Q_samples, I_symbols, Q_symbols,
                                       modulation_type, sampling_rate, 
                                       samples_per_symbol):
    
    create_figure_with_subplots(3, 3, figsize=(15, 12))
    
    // Row 1: Time domain
    select_subplot(1, 1)
    plot_time_domain_signals(I_samples, Q_samples, sampling_rate)
    
    select_subplot(1, 2)
    plot_magnitude_and_phase(I_samples, Q_samples, sampling_rate)
    
    select_subplot(1, 3)
    plot_spectrum(I_samples, Q_samples, sampling_rate)
    
    // Row 2: Symbol-level analysis
    select_subplot(2, 1)
    plot_constellation_diagram(I_symbols, Q_symbols, modulation_type)
    
    select_subplot(2, 2)
    plot_eye_diagram(I_samples, samples_per_symbol, channel='I')
    
    select_subplot(2, 3)
    plot_eye_diagram(Q_samples, samples_per_symbol, channel='Q')
    
    // Row 3: Quality metrics
    select_subplot(3, 1)
    if modulation_type contains "QAM":
        plot_evm_vs_time(I_symbols, Q_symbols, get_constellation(modulation_type))
    else:
        plot_phase_trajectory(I_samples, Q_samples, sampling_rate)
    
    select_subplot(3, 2)
    if modulation_type == "FSK":
        plot_spectrogram(I_samples, Q_samples, sampling_rate)
    else:
        plot_iq_trajectory(I_samples, Q_samples, 50)  // 50 symbols
    
    select_subplot(3, 3)
    plot_timing_error_or_statistics(I_symbols, Q_symbols)
    
    suptitle(modulation_type + " Demodulation Analysis Dashboard")
    tight_layout()
end function
```
## Quick Reference: Best Visualisation by Question
**"Is my signal getting through?"**
- Constellation diagram (any modulation)
- Received power vs time

**"Do I have timing synchronization?"**
- Eye diagram (PSK, QAM)
- Zero-crossing plot (BPSK)

**"Do I have carrier synchronization?"**
- Constellation diagram rotation over time
- Phase trajectory plot
- Frequency offset plot

**"What's my link quality?"**
- EVM vs time (QAM)
- BER vs SNR curve
- Constellation tightness

**"Are my frequencies correct?"**
- Spectrogram (FSK)
- FFT magnitude
- Instantaneous frequency plot

**"Is there inter-symbol interference?"**
- Eye diagram (all modulations)
- Eye width and height measurements

**"What's the spectral efficiency?"**
- Power spectral density
- Occupied bandwidth plot
- Spectral mask compliance