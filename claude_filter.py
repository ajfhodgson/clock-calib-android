# Thanks to Claude AI: https://claude.ai/chat/fc57eabe-0c10-4001-94e9-a0e785106ac0
import numpy as np
from scipy import signal

class ClockTickDetector:
    """Stateful clock tick detector for continuous audio streams."""
    
    def __init__(self, sr=44100, min_tick_interval=0.3):
        self.sr = sr
        self.min_tick_interval = min_tick_interval
        self.min_samples = int(min_tick_interval * sr)
        
        # Design filters once
        self.sos = signal.butter(4, [800, 8000], 'bandpass', fs=sr, output='sos')
        
        # Initialize filter states
        self.sos_state = signal.sosfilt_zi(self.sos)
        
        # Initialize envelope filter states
        self.fast_tc = 0.005
        self.slow_tc = 0.15
        self.alpha_fast = 1 - np.exp(-1/(sr * self.fast_tc))
        self.alpha_slow = 1 - np.exp(-1/(sr * self.slow_tc))
        
        # Filter coefficients for envelope followers
        self.b_fast = [self.alpha_fast]
        self.a_fast = [1, self.alpha_fast - 1]
        self.b_slow = [self.alpha_slow]
        self.a_slow = [1, self.alpha_slow - 1]
        
        # Initialize envelope filter states to zero
        self.fast_state = signal.lfilter_zi(self.b_fast, self.a_fast) * 0
        self.slow_state = signal.lfilter_zi(self.b_slow, self.a_slow) * 0
        
        # Track last detection time to prevent duplicates across chunks
        self.last_tick_sample = -self.min_samples
        self.total_samples_processed = 0
        
        # For adaptive threshold - keep running statistics
        self.onset_history = []
        self.max_history_samples = sr * 10  # Keep 10 seconds of history
        
    def process_chunk(self, audio_chunk):
        """
        Process a chunk of audio and return tick times.
        
        Parameters:
        -----------
        audio_chunk : numpy array of audio samples
        
        Returns:
        --------
        tick_times : array of tick times in seconds (absolute time from reset)
        debug_info : dict with diagnostic arrays and time axis
        """
        
        # Create time axis for this chunk
        chunk_start_time = self.total_samples_processed / self.sr
        time_axis = np.arange(len(audio_chunk)) / self.sr + chunk_start_time
        
        # Step 1: Bandpass filter (with state)
        filtered, self.sos_state = signal.sosfilt(
            self.sos, audio_chunk, zi=self.sos_state
        )
        
        # Step 2: Rectify
        rectified = np.abs(filtered)
        
        # Step 3: Fast envelope (with state)
        fast_env, self.fast_state = signal.lfilter(
            self.b_fast, self.a_fast, rectified, zi=self.fast_state
        )
        
        # Step 4: Slow envelope (with state)
        slow_env, self.slow_state = signal.lfilter(
            self.b_slow, self.a_slow, rectified, zi=self.slow_state
        )
        
        # Step 5: Onset strength
        onset_strength = fast_env - slow_env
        onset_strength = np.maximum(onset_strength, 0)
        
        # Step 6: Update onset history for adaptive threshold
        self.onset_history.extend(onset_strength.tolist())
        if len(self.onset_history) > self.max_history_samples:
            self.onset_history = self.onset_history[-self.max_history_samples:]
        
        # Step 7: Adaptive threshold using history
        median_onset = np.median(self.onset_history)
        mad = np.median(np.abs(np.array(self.onset_history) - median_onset))
        threshold = median_onset + 3 * mad
        
        # Step 8: Find peaks in this chunk
        peaks, properties = signal.find_peaks(
            onset_strength,
            height=threshold,
            prominence=threshold * 0.3
        )
        
        # Step 9: Filter out peaks too close to last tick from previous chunk
        valid_peaks = []
        for peak in peaks:
            global_peak_sample = self.total_samples_processed + peak
            if global_peak_sample - self.last_tick_sample >= self.min_samples:
                valid_peaks.append(peak)
                self.last_tick_sample = global_peak_sample
        
        valid_peaks = np.array(valid_peaks)
        
        # Convert to absolute time (seconds from reset)
        tick_times = (self.total_samples_processed + valid_peaks) / self.sr if len(valid_peaks) > 0 else np.array([])
        
        # Update total samples processed
        self.total_samples_processed += len(audio_chunk)
        
        debug_info = {
            'onset_strength': onset_strength,
            'threshold': threshold,
            'fast_env': fast_env,
            'slow_env': slow_env,
            'filtered': filtered,
            'time_axis': time_axis  # Time in seconds for each sample in chunk
        }
        
        return tick_times, debug_info
    
    def reset(self):
        """Reset all states (use when starting a new recording)."""
        self.sos_state = signal.sosfilt_zi(self.sos)
        self.fast_state = signal.lfilter_zi(self.b_fast, self.a_fast) * 0
        self.slow_state = signal.lfilter_zi(self.b_slow, self.a_slow) * 0
        self.last_tick_sample = -self.min_samples
        self.total_samples_processed = 0
        self.onset_history = []


# Usage example:
#detector = ClockTickDetector(sr=44100, min_tick_interval=0.3)

# Process continuous stream
#for chunk in audio_stream:  # Each chunk is 4 seconds
#    tick_times, debug_info = detector.process_chunk(chunk)
#    
#    # tick_times are now in absolute seconds from the first chunk
#    print(f"Found {len(tick_times)} ticks at times: {tick_times}")
#    
# When starting a new recording:
#detector.reset()
