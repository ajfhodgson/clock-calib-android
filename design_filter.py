from scipy import signal
import numpy as np
import json

sr = 44100  # your sample rate

# Design the bandpass filter
sos = signal.butter(4, [800, 8000], 'bandpass', fs=sr, output='sos')

# Envelope filter coefficients
fast_tc = 0.005
slow_tc = 0.15
alpha_fast = 1 - np.exp(-1/(sr * fast_tc))
alpha_slow = 1 - np.exp(-1/(sr * slow_tc))

# Save to JSON
coeffs = {
    'sos': sos.tolist(),
    'alpha_fast': alpha_fast,
    'alpha_slow': alpha_slow
}

with open('filter_coeffs.json', 'w') as f:
    json.dump(coeffs, f, indent=2)

print("Filter coefficients saved!")