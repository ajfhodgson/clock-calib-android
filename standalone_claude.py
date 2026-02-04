import numpy as np
from scipy import signal
import librosa
import matplotlib.pyplot as plt
import csv
import os

clock_name = 'Djd mantel'
clock_name = 'Long case'
clock_name = 'Box clock'

plot_each_chunk = False
mad_factor = 2.0  # Threshold sensitivity factor

# idea - having calculated histograms, 
# identify two highest bins of dt1, identify all bins that have counts above 5% of 
# the average count of the two biggest bins. 
# ticks falling inside these bin ranges are 'valid' ticks, those outside, 'invalid' ticks
# remove form the ticks array those ticks that are invalid, recalculate dt1 and dt2
# what about removing some invalid ticks might make others valid (ones with dt2 = tick interval)?
# calculate average of dt1s that fall inside these bins - that's the estimated tick interval
# calculate average dt1 of ticks falling above and below the median of the valid ticks => beat error
# report how long this calculation and pricessing takes! Will get progressinvely longer until
# some max window (15 mins?) is reached, then stabilise.

# discard as invalid any 'ticks' whose dt1 falls outside these bins
# consider 'valid' ticks only those whose dt1 falls inside these two bins
# rerun the histogram?

class ClockTickDetector:
    """Stateful clock tick detector for continuous audio streams."""
    
    def __init__(self, sr=44100, min_tick_interval=0.1):
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
        # Filter out zeros (silence) to calculate statistics on actual signal content
        history_arr = np.array(self.onset_history)
        valid_history = history_arr[history_arr > 1e-9]
        
        if len(valid_history) > 0:
            median_onset = np.median(valid_history)
            mad = np.median(np.abs(valid_history - median_onset))
            threshold = median_onset + (mad_factor * mad)
        else:
            threshold = 0.0
        
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
            'time_axis': time_axis,  # Time in seconds for each sample in chunk
            'audio_chunk': audio_chunk  # Include original audio for plotting
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


def append_tick_times(new_ticks, previous_ticks):
    """Appends tick times to claude_ticks.csv with dt1 and dt2 columns."""
    if len(new_ticks) == 0:
        return

    filename = clock_name + "_ticks.csv"
    try:
        file_exists = os.path.exists(filename)
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Tick_Time_s", "dt1", "dt2"])
            
            # Combine last 2 previous ticks with new ticks for context
            context = list(previous_ticks[-2:]) if len(previous_ticks) > 0 else []
            combined = context + list(new_ticks) # 2 old plus all of new
            offset = len(context) # typically 2

            for i, tick in enumerate(new_ticks):
                idx = i + offset # i is index into new_ticks, idx is index into combined
                dt1 = combined[idx] - combined[idx-1] if idx >= 1 else ""
                dt2 = combined[idx] - combined[idx-2] if idx >= 2 else ""
                writer.writerow([tick, dt1, dt2])
    except Exception as e:
        print(f"I/O Error (append ticks): {e}")

def plot_chunk_results(chunk_num, total_chunks, tick_times, debug_info):
    """Plot the results for a single chunk."""
    time_axis = debug_info['time_axis']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot audio in grey
    ax.plot(time_axis, debug_info['audio_chunk'], color='grey', alpha=0.5, label='Audio', linewidth=0.5)
    
    # Plot filtered in blue
    ax.plot(time_axis, debug_info['filtered'], color='blue', alpha=0.6, label='Filtered', linewidth=0.8)
    
    # Plot fast and slow envelopes in purple
    ax.plot(time_axis, debug_info['fast_env'], color='purple', alpha=0.8, label='Fast envelope', linewidth=1)
    ax.plot(time_axis, debug_info['slow_env'], color='purple', alpha=0.5, linestyle='--', label='Slow envelope', linewidth=1)
    
    # Plot onset strength in green
    ax.plot(time_axis, debug_info['onset_strength'], color='green', alpha=0.8, label='Onset strength', linewidth=1.5)
    
    # Plot threshold as horizontal line
    ax.axhline(debug_info['threshold'], color='orange', linestyle=':', label='Threshold', linewidth=1)
    
    # Plot detected ticks in red
    if len(tick_times) > 0:
        # Find the y-values for the ticks on the onset strength curve
        tick_indices = np.searchsorted(time_axis, tick_times)
        tick_indices = np.clip(tick_indices, 0, len(debug_info['onset_strength']) - 1)
        tick_y_values = debug_info['onset_strength'][tick_indices]
        
        ax.scatter(tick_times, tick_y_values, color='red', s=100, zorder=5, marker='o', label=f'Ticks ({len(tick_times)})')
        
        # Add vertical lines at tick positions
        for tick_time in tick_times:
            ax.axvline(tick_time, color='red', alpha=0.3, linestyle='-', linewidth=1)
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title(f'Chunk {chunk_num}/{total_chunks} - Clock Tick Detection', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Attempt to maximize the plot window
    try:
        manager = plt.get_current_fig_manager()
        backend = plt.get_backend().lower()
        if 'tk' in backend:
            if os.name == 'nt':
                manager.window.state('zoomed')
            else:
                manager.resize(*manager.window.maxsize())
        elif 'qt' in backend:
            manager.window.showMaximized()
        elif 'wx' in backend:
            manager.frame.Maximize(True)
    except Exception as e:
        print(f"Window maximization failed: {e}")

    plt.tight_layout()
    plt.show()


def plot_intervals_histogram(counts1, bins1, counts2, bins2):
    """Plot histograms for dt1 and dt2 using pre-calculated counts in separate subplots."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot dt1
    ax = axs[0]
    bin_widths = np.diff(bins1)
    bin_centers = bins1[:-1] + bin_widths / 2
    rects = ax.bar(bin_centers, counts1, width=bin_widths*0.9, 
                   color='skyblue', edgecolor='black', label='dt1 (Tick)')
    
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    bin_span_ms = bin_widths[0] * 1000 if len(bin_widths) > 0 else 0
    total_ticks = np.sum(counts1)
    ax.set_title(f'{clock_name} - Tick Interval\nBins: {len(bins1)-1}, Bin Span: {bin_span_ms:.2f} ms, Total: {total_ticks} ticks, mad_factor: {mad_factor}', fontsize=12)
    ax.set_xlabel('Seconds')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    if len(bins1) > 1:
        ax.set_xlim(0, bins1[-1])

    # Plot dt2
    ax = axs[1]
    bin_widths = np.diff(bins2)
    bin_centers = bins2[:-1] + bin_widths / 2
    rects = ax.bar(bin_centers, counts2, width=bin_widths*0.9, 
                   color='lightgreen', edgecolor='black', label='dt2 (2-Tick)')
    
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    bin_span_ms = bin_widths[0] * 1000 if len(bin_widths) > 0 else 0
    ax.set_title(f'{clock_name} - 2-Tick Interval\nBins: {len(bins1)-1}, Bin Span: {bin_span_ms:.2f} ms, Total: {total_ticks} ticks, mad_factor: {mad_factor}', fontsize=12)
    ax.set_xlabel('Seconds')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    if len(bins2) > 1:
        ax.set_xlim(0, bins2[-1])

    plt.tight_layout()
    plt.show()

# Usage example with m4a file
if __name__ == "__main__":

    # Load m4a file and resample to 44100 Hz
    print("Loading audio file...")
    audio, sr = librosa.load(clock_name + ".wav", sr=44100, mono=True)
    print(f"Loaded {len(audio)/sr:.2f} seconds of audio at {sr} Hz")
    
    # Create chunks of 4 seconds
    chunk_size = 4 * sr  # 4 seconds * 44100 samples/sec
    chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]
    print(f"Split into {len(chunks)} chunks of ~4 seconds each\n")
    
    # Create detector and process chunks
    detector = ClockTickDetector(sr=44100)
    
    # Replace previous CSV if exists
    if os.path.exists(clock_name +"ticks.csv"):
        os.remove(clock_name +"ticks.csv")

    all_tick_times = [] # entire recording - all chunks
    for i, chunk in enumerate(chunks):
        tick_times, debug_info = detector.process_chunk(chunk)
        append_tick_times(tick_times, all_tick_times) # appends to ticks file (with dt1 and dt2)
        all_tick_times.extend(tick_times) # add it to the array itself
        
        print(f"\nChunk {i+1}/{len(chunks)}: "
              f"{debug_info['time_axis'][0]:.2f}s to {debug_info['time_axis'][-1]:.2f}s - "
              f"found {len(tick_times)} ticks, total so far: {len(all_tick_times)}"
              )
        if len(tick_times) > 0:
            print(f"Tick times: {tick_times}") 
        
        if plot_each_chunk :
            plot_chunk_results(i+1, len(chunks), tick_times, debug_info)

    #end for

    # Calculate (and plot) histograms
    if len(all_tick_times) >= 3:
        ticks = np.array(all_tick_times)
        dt1 = np.diff(ticks)
        dt2 = ticks[2:] - ticks[:-2]
        
        counts1, bins1 = np.histogram(dt1, bins=50)
        counts2, bins2 = np.histogram(dt2, bins=50)
        
        plot_intervals_histogram(counts1, bins1, counts2, bins2)
    else:
        print("Not enough ticks for histogram analysis")

    print("\nFine.")
# end main
