from sample_bpsk import generate_bpsk_I_Q_baseband
from sample_qpsk import generate_qpsk_I_Q_baseband
from passband import generate_passband
from visualisation import plot_time_domain, plot_constellation, eye_diagram, plot_spectrogram
import sys
import argparse

CARRIER_FREQUENCY = 10000  # in Hz
SAMPLING_RATE = 100000  # in Hz
SYMBOL_RATE = 1000  # in symbols per second
DURATION = 1  # in seconds


def generate_signal(args):
    if args.modulation == 'BPSK':
        signal_type = 'BPSK'
        I_baseband, Q_baseband = generate_bpsk_I_Q_baseband(sampling_rate=args.sampling_rate, symbol_rate=args.symbol_rate, duration=args.duration)
    elif args.modulation == 'QPSK':
        signal_type = 'QPSK'
        I_baseband, Q_baseband = generate_qpsk_I_Q_baseband(sampling_rate=args.sampling_rate, symbol_rate=args.symbol_rate, duration=args.duration)
    time_axis, passband_signal = generate_passband(I_baseband, Q_baseband, args.sampling_rate, args.duration, args.carrier_frequency)
    
    
    samples_per_symbol = args.sampling_rate // args.symbol_rate
    I_symbols = I_baseband[::samples_per_symbol]
    Q_symbols = Q_baseband[::samples_per_symbol]  
    if args.visualisation =='time_domain':
        plot_time_domain(signal_type, time_axis, I_baseband, Q_baseband, passband_signal)
    elif args.visualisation == 'constellation':
        plot_constellation(I_symbols, Q_symbols)
    elif args.visualisation == 'eye_diagram':
        eye_diagram(passband_signal, samples_per_symbol)
    elif args.visualisation == 'spectrogram':
        plot_spectrogram(passband_signal, args.sampling_rate)
            
def main():
    modulations = ['bpsk', 'qpsk']
    visualisations = ['time_domain', 'constellation', 'eye_diagram', 'spectrogram']
    parser = argparse.ArgumentParser(
        description=f"Generate and visualize {modulations} signals",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog='''
Example usage:
python main.py --modulation BPSK --duration 2 --sampling_rate 100000 --symbol_rate 1000 --carrier_frequency 100000,--visualisation time_domain\n
python main.py --modulation QPSK --duration 1 --sampling_rate 200000 --symbol_rate 2000 --carrier_frequency 50000
        '''
    )
    parser.add_argument('-m', '--modulation', type=str, choices=modulations, required=True, 
                        help='Type of modulation to generate (BPSK or QPSK)')
    parser.add_argument('-d', '--duration', type=int, default=DURATION, help='Duration of the signal in seconds')
    parser.add_argument('-sr', '--sampling_rate', type=int, default=SAMPLING_RATE, help='Sampling rate in Hz')
    parser.add_argument('-fr', '--symbol_rate', type=int, default=SYMBOL_RATE, help='Symbol rate in symbols per second')
    parser.add_argument('-cf', '--carrier_frequency', type=int, default=CARRIER_FREQUENCY, help='Carrier frequency in Hz')
    parser.add_argument('-v', '--visualisation', type=str, choices=visualisations,  default='time_domain',
                        help='Types of visualisations to generate') #removed nargs='+', as it was causing an error when trying to run
    args = parser.parse_args()
    args.modulation = args.modulation.upper()
    
    generate_signal(args)
    # if isinstance(args.visualisation, str):
    #     args.visualisation = [args.visualisation]
      
if __name__ == "__main__":
    main()