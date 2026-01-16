from sample_bpsk import generate_bpsk_I_Q_baseband
from sample_qpsk import generate_qpsk_I_Q_baseband
from passband import generate_passband
from visualisation import plot_time_domain
import sys

CARRIER_FREQUENCY = 10000  # in Hz
SAMPLING_RATE = 100000  # in Hz
SYMBOL_RATE = 1000  # in symbols per second
DURATION = 1  # in seconds

def main(args):
    if not args[1:]:
        print("Please specify modulation type: 'BPSK' or 'QPSK'")
        sys.exit(1)
        
        
    modulation_type = args[1].upper()
    
    if modulation_type == 'BPSK':
        signal_type = 'BPSK'
        I_baseband, Q_baseband = generate_bpsk_I_Q_baseband(sampling_rate=SAMPLING_RATE, symbol_rate=SYMBOL_RATE, duration=DURATION)
    elif modulation_type == 'QPSK':
        signal_type = 'QPSK'
        I_baseband, Q_baseband = generate_qpsk_I_Q_baseband(sampling_rate=SAMPLING_RATE, symbol_rate=SYMBOL_RATE, duration=DURATION)
    time_axis, passband_signal = generate_passband(I_baseband, Q_baseband, SAMPLING_RATE, DURATION, CARRIER_FREQUENCY)
    plot_time_domain(signal_type, time_axis, I_baseband, Q_baseband, passband_signal)
    
if __name__ == "__main__":
    main(sys.argv)