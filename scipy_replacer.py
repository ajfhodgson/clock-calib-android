# This is a stub intended to replace the scipy library function, to avoid
# the fortram dependencies. See https://claude.ai/share/c6779fb4-87a9-4bde-94da-93839a481944

import numpy as np

class ScipyReplacer:
    def __init__(self, sr):
        # Load coefficients for 
        # 1. a Buterworth second order filter, bandpass from 800 to 8000 Hz (typical clock tick frequencies)
        # coefficients taken from offline program 'design_filter.py' which depends on scipy
        # self.sos = signal.butter(4, [800, 8000], 'bandpass', fs=sr, output='sos')

        self.sos = np.array([ 
            [ 0.024406700852204526, 0.04881340170440905, 0.024406700852204526, 1.0, -0.6249181246041945, 0.1487578690348958 ],
            [ 1.0, 2.0, 1.0, 1.0, -0.670658858820238, 0.5466134300303429 ],
            [ 1.0, -2.0, 1.0, 1.0, -1.7723326172388825, 0.7884484792733465 ],
            [ 1.0, -2.0, 1.0, 1.0, -1.914304839059571, 0.9273030296653866  ]
        ])
        
        # Initialize states
        n_sections = self.sos.shape[0]
        self.sos_state = np.zeros((n_sections, 2))
    
    def reset(self):
        """Reset all states."""
        n_sections = self.sos.shape[0]
        self.sos_state = np.zeros((n_sections, 2))
        self.last_tick_sample = -self.min_samples
        self.sample_count_before_this_chunk = 0
        self.onset_history = []

    def sosfilt(self, audio_chunk, zi=None):
        """Apply second-order sections filter (replaces signal.sosfilt)."""
        if zi is None:
            zi = np.zeros((self.sos.shape[0], 2))
        
        y = audio_chunk.copy()
        new_zi = np.zeros_like(zi)
        
        for i, section in enumerate(self.sos):
            b0, b1, b2, a0, a1, a2 = section
            
            # Direct Form II Transposed implementation
            y_new = np.zeros_like(y)
            z1, z2 = zi[i]
            
            for n in range(len(y)):
                y_new[n] = b0 * y[n] + z1
                z1 = b1 * y[n] - a1 * y_new[n] + z2
                z2 = b2 * y[n] - a2 * y_new[n]
            
            new_zi[i] = [z1, z2]
            y = y_new
        
        return y, new_zi

    
#def lfilter(self, alpha, x, state):       self is if it's an instance function inside the class
def lfilter(alpha, x, state):
    """
    Apply single-pole IIR lowpass filter: y[n] = alpha*x[n] + (1-alpha)*y[n-1]
    
    Args:
        alpha: Filter coefficient (0 < alpha < 1)
        x: Input signal array
        state: Filter state array [1] element
    
    Returns:
        y: Filtered output
        state: Updated state (modified in-place, but also returned for scipy compatibility)
    """
    y = np.zeros_like(x)
    z = state[0]
    
    for n in range(len(x)):
        y[n] = alpha * x[n] + (1 - alpha) * z
        z = y[n]
    
    state[0] = z
    return y, state


#def find_peaks(self, x, height, prominence):
def find_peaks(x, height, prominence):
    """Simple peak detection (replaces signal.find_peaks)."""
    peaks = []
    
    for i in range(1, len(x) - 1):
        # Check if it's a local maximum
        if x[i] > x[i-1] and x[i] > x[i+1] and x[i] >= height:
            # Check prominence (simplified)
            left_min = np.min(x[max(0, i-10):i])
            right_min = np.min(x[i+1:min(len(x), i+11)])
            prom = x[i] - max(left_min, right_min)
            
            if prom >= prominence:
                peaks.append(i)
    
    return np.array(peaks), {}

#def find_peaks_2(self, x, height=None, prominence=None):
def find_peaks_2(x, height=None, prominence=None):
    """Peak detection - closer to scipy's algorithm."""
    peaks = []
    
    # Find local maxima
    for i in range(1, len(x) - 1):
        if x[i] > x[i-1] and x[i] > x[i+1]:
            if height is None or x[i] >= height:
                peaks.append(i)
    
    if prominence is None:
        return np.array(peaks), {}
    
    # Calculate prominence for each peak
    filtered_peaks = []
    for peak in peaks:
        # Find lowest point between this peak and higher peaks on each side
        # Simplified: search until we hit a higher peak or edge
        
        # Search left
        left_min = x[peak]
        for j in range(peak - 1, -1, -1):
            if x[j] > x[peak]:  # Found higher peak
                break
            left_min = min(left_min, x[j])
        
        # Search right
        right_min = x[peak]
        for j in range(peak + 1, len(x)):
            if x[j] > x[peak]:  # Found higher peak
                break
            right_min = min(right_min, x[j])
        
        # Prominence is height above the higher of the two bases
        prom = x[peak] - max(left_min, right_min)
        
        if prom >= prominence:
            filtered_peaks.append(peak)
    
    return np.array(filtered_peaks), {}