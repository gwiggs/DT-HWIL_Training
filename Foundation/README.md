### Week 1: Digital Twin Concepts & DSP Fundamentals

#### Learning Objectives
- Understand digital twin taxonomy and architectures
- Review core DSP concepts for EW applications
- Set up development environment

#### Exercises

**Exercise 1.1: Environment Validation** (2 hours)
```python
# Validate your Python setup
# Create a script that:
# - Generates a sine wave at 1 kHz sampled at 44.1 kHz
# - Plots time domain representation
# - Computes and plots FFT
# - Saves results to file
```

**Exercise 1.2: Complex Signal Basics** (3 hours)
```python
# Implement from scratch (no scipy.signal):
# - Complex exponential signal generation
# - I/Q representation of modulated signals
# - Amplitude and phase extraction
# - Visualization of constellation diagrams
```

**Exercise 1.3: Sampling Theory** (2 hours)
```python
# - Demonstrate aliasing with various sampling rates
# - Implement and visualize the sampling theorem
# - Create anti-aliasing filter example
```

#### Guided Project 1: Signal Analysis Toolkit (5 hours)

```python
# Build a Python module that provides:
# - Signal generation functions (sine, square, triangle, noise)
# - Time-domain analysis (RMS, peak detection, zero crossings)
# - Frequency-domain analysis (FFT, PSD, spectrogram)
# - Basic filtering (moving average, simple IIR)
# - Visualization utilities
```
---

### Week 2: Advanced DSP for EW Applications

#### Learning Objectives
- Understand modulation schemes used in EW
- Implement digital filters
- Process complex baseband signals

#### Exercises

**Exercise 2.1: Filter Design** (4 hours)
```python
# Design and implement:
# 1. Low-pass FIR filter using window method
# 2. Band-pass IIR filter (Butterworth)
# 3. Notch filter for interference rejection
# Compare frequency responses and group delay
```

**Exercise 2.2: Modulation Implementation** (4 hours)
```python
# Implement modulators/demodulators for:
# - BPSK
# - QPSK
# - 16-QAM
# Include symbol timing recovery and carrier synchronization
```

**Exercise 2.3: Pulse Detection** (3 hours)
```python
# - Implement matched filter for pulse detection
# - Add noise and measure detection probability
# - Plot ROC curves for different SNR levels
```

#### Guided Project 2: EW Signal Simulator (6 hours)
```python
# Create a Python application that simulates common EW signals:
# - Pulsed radar (adjustable PRF, pulse width, frequency)
# - Continuous wave (CW) jamming
# - Swept frequency jamming
# - Communication signals (AM, FM)
```
---

### Week 3: Digital Twin Architecture for Defense Systems

#### Learning Objectives
- Understand digital twin maturity levels
- Design system architectures for EW digital twins
- Learn real-time constraints and requirements

#### Exercises

**Exercise 3.1: System Decomposition** (3 hours)
```python
# - Choose an EW system (e.g., radar warning receiver)
# - Create functional block diagram
# - Identify interfaces and data flows
# - Document timing requirements
# - Propose digital twin architecture
```

**Exercise 3.2: Requirements Analysis** (3 hours)
```python
# - Define digital twin fidelity requirements
# - Specify latency budgets for HWIL
# - Identify state variables to monitor
# - Create traceability matrix
```

**Exercise 3.3: Interface Design** (3 hours)
```python
# Design a message protocol for digital twin communication:
# - Define message structures (using Python dataclasses)
# - Implement serialization/deserialization
# - Add checksum/validation
# - Create protocol documentation
```

#### Unguided Project 3: Digital Twin Design Document (8 hours)

**Scenario**: Design a digital twin for a radar warning receiver (RWR) system
```python
Create a comprehensive design document including:
1. System overview and operational context
2. Digital twin architecture (physical, virtual, connection layers)
3. Interface specifications
4. Fidelity requirements and validation approach
5. HWIL integration strategy
6. Data flow diagrams
7. Performance requirements (latency, throughput)
8. Test scenarios and success criteria
```
---
