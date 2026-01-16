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
