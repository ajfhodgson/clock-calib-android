import numpy as np
from scipy import signal
from kivy.utils import platform

if platform == 'win':
    import windows_plotting  # keep matplotlib separate

class ClockBeatDetector:
    """Stateful clock tick detector for continuous audio streams."""
    
    def __init__(self, sr=44100, min_tick_interval=0.1):
        self.sr = sr
        self.min_tick_interval = min_tick_interval # not sure that this is useful
        self.min_samples = int(min_tick_interval * sr)
        
        # Design bandpass filter - butterworth bandpass from 800 to 8000 Hz (typical clock tick frequencies)
        self.sos = signal.butter(4, [800, 8000], 'bandpass', fs=sr, output='sos') # second order filter
        self.sos_state = signal.sosfilt_zi(self.sos)         # Initialize filter states
        
        # Run a fast and a slow tracker, to detect 'onset' edge of tick
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

        self.mad_factor = 1.0
        
        # Track last detection time to prevent duplicates across chunks
        self.last_tick_sample = -self.min_samples
        self.sample_count_before_this_chunk = 0
        
        # For adaptive threshold - keep running statistics
        self.onset_history = []
        self.max_history_samples = sr * 10  # Keep 10 seconds of history

    
    def reset(self):
        """Reset all states (use when starting a new recording)."""
        self.sos_state = signal.sosfilt_zi(self.sos)
        self.fast_state = signal.lfilter_zi(self.b_fast, self.a_fast) * 0
        self.slow_state = signal.lfilter_zi(self.b_slow, self.a_slow) * 0
        self.last_tick_sample = -self.min_samples
        self.sample_count_before_this_chunk = 0
        self.onset_history = []


    def process_chunk(self, audio_chunk):
        # takes [4] seconds of audio data, identifies rising _edges_ in level that rise quickly above an adaptive threshold, 
        # returns their times in seconds (absolute time since ClockBeatDetector.reset()) 
        # also returns a dict of diagnostic arrays for plotting, etc

        # Create time axis for this chunk
        chunk_start_time = self.sample_count_before_this_chunk / self.sr
        time_axis = np.arange(len(audio_chunk)) / self.sr + chunk_start_time
        
        # 2-pole Butterworth band-pass filter
        filtered, self.sos_state = signal.sosfilt(self.sos, audio_chunk, zi=self.sos_state)  # Bandpass filter (with state)
        rectified = np.abs(filtered) # Rectify to get envelope shape
        fast_env, self.fast_state = signal.lfilter( self.b_fast, self.a_fast, rectified, zi=self.fast_state )
        slow_env, self.slow_state = signal.lfilter( self.b_slow, self.a_slow, rectified, zi=self.slow_state )
        onset_strength = fast_env - slow_env  # edge detector array of values, times matching audio_chunk and time_axis
        onset_strength = np.maximum(onset_strength, 0) # discard any negative values in the array
        
        # Update onset history for adaptive threshold - the most recent self.max_history_samples [10s] seconds of data
        self.onset_history.extend(onset_strength.tolist()) # entire history since initialisation of ClockBeatDetector class
        if len(self.onset_history) > self.max_history_samples: 
            self.onset_history = self.onset_history[-self.max_history_samples:] # truncate to most recent [10] seconds of data, to adapts to recent conditions
        
        # Calculate adaptive threshold to test onset against, using history - median + mad_factor * MAD 
        history_arr = np.array(self.onset_history)
        valid_history = history_arr[history_arr > 1e-9] # ignore near-zero values to avoid skewing median and MAD
        
        if len(valid_history) > 0:
            median_onset = np.median(valid_history) # halfway between max and minimum of the recent onset strength values
            mad = np.median(np.abs(valid_history - median_onset))
            threshold = median_onset + (self.mad_factor * mad)
        else:
            threshold = 0.0
        
        # Find peaky edges in this chunk - there will be false positives (and false negatives)!
        peak_indexes, properties = signal.find_peaks(
            onset_strength,
            height=threshold, # Minimum height of peaks. Peaks below this value are ignored.
            prominence=threshold * 0.3 # Minimum prominence of peaks, how much a peak stands out from its surroundings
        )
        
        found_peak_indexes = []
        for peak_index in peak_indexes: # peak_index is index of the peak in the current chunk
            global_peak_index = self.sample_count_before_this_chunk + peak_index
            if global_peak_index - self.last_tick_sample >= self.min_samples:
                found_peak_indexes.append(peak_index) # still index of the peak in the current chunk, not global
                self.last_tick_sample = global_peak_index
        found_peak_indexes = np.array(found_peak_indexes) 
        
        # Convert to absolute time (seconds from reset) for this chunk's rising edges
        chunk_edge_times = (self.sample_count_before_this_chunk + found_peak_indexes) / self.sr if len(found_peak_indexes) > 0 else np.array([])

        # Update total samples processed to keep track of absolute time across chunks
        self.sample_count_before_this_chunk += len(audio_chunk)
        
        time_series_data = {
            'time_axis': time_axis,  # Time in seconds for each sample in chunk
            'audio_chunk': audio_chunk,  # Include original audio for plotting
            'onset_strength': onset_strength,
            'threshold': np.full_like(time_axis, threshold),
            'fast_env': fast_env,
            'slow_env': slow_env,
            'filtered': filtered,
        }
        
        return time_series_data, chunk_edge_times

# ======= end of process_chunk method ===========================================

    def weed_edges_in_window(self, edge_times, plot_it, clock_name):
        # called when we have a decent window's-worth of edges, enough to make a useful histogram
        # I expect after 4s, 16s, 64s, 256s, 1024s, 4096s, 16384s (4.5 hours) - will get more accurate as more ticks are added, but takes longer to calculate and plot, and less responsive to changes in tick interval over time.

        # calculate dt1 and dt2 and calculate histograms for them to weed out false positives (non-beat edges)
        beats = np.array(edge_times) # start optimistically assuming all edges are beats, will weed out the false positives using the histograms of dt1 and dt2
        dt1 = np.concatenate(([0], np.diff(beats)))
        dt2 = np.concatenate(([0, 0], beats[2:] - beats[:-2]))    
        counts1, bins1 = np.histogram(dt1, bins=50) # &&ToDo - should this be fixed or auto? 
        counts2, bins2 = np.histogram(dt2, bins=50) # just for visual histogram, not used 
        # &&ToDo - worry that the histograms will include the padding zeros at the start of dt1 and dt2, which will skew the histograms slightly

        # find mid-point between the two most-populated bins of dt1
        indexes_of_biggest_bins = np.argsort(counts1)[-2:]
        idx_low, idx_high = np.sort(indexes_of_biggest_bins)
        crude_beat_period = (bins1[idx_low] + bins1[idx_high + 1]) / 2
#        print(f"Crude beat period estimated from histogram: {crude_beat_period:.3f} seconds (bins {idx_low} and {idx_high}, counts {counts1[idx_low]} and {counts1[idx_high]})")

        # weed out false positive beats where the next beat is a more plausible beat that this one
        bad_beats = []
        for i in range(2, len(beats)-2): # we've only got dt2 values after i=2 
            # so we'll have to give the first two beats a pass, as we don't have diffs backwards for them
            # we're considering beat[i] for rejection. We look at how long after beat[i-1] it was (dt1[i])
            # and also at beat[i+1] and how long that was after beat[i-1]. 
            # if beat[i+1] is closer to the crude beat period than beat[i], then beat[i] is likely to be a false positive, 
            # and we will erase it from beats[]
    #        print(f"Checking beat {i} at time {beats[i]:.2f}s: dt1={dt1[i]:.3f}s, dt2={dt2[i+1]:.3f}s, crude_beat_period={crude_beat_period:.3f}s")
            if (abs(dt1[i] - crude_beat_period) > abs(dt2[i+1] - crude_beat_period)) : # i is BAD - erase it!
                bad_beats.append(beats[i]) # keep track of which beats we are erasing, for reporting and debugging
                dt1[i+1] = dt1[i+1] + dt1[i] # update dt1 of the next beat to skip the erased edge
                dt2[i+2] = dt2[i+2] + dt1[i] # update dt2 of the next beat to skip the erased edge
                beats[i] = dt1[i] = dt2[i] = np.nan # this beat is erased, so its time, dt1 and dt2 are now meaningless
    #            print(f"  => Rejected beat {i} at time {beats[i]:.2f}s as false positive, dt1={dt1[i]:.3f}s is further from crude beat period than dt2={dt2[i+1]:.3f}s")
            else:
    #            print(f"  => Accepted beat {i} at time {beats[i]:.2f}s as valid, dt1={dt1[i]:.3f}s is closer to crude beat period than dt2={dt2[i+1]:.3f}s")
                pass

        # now calculate histograms for the remaining beats, to report on the distribution of dt1 and dt2 for the 'good' beats, and to plot the histograms with the bad beats removed

        good_beats = beats[~np.isnan(beats)]
        good_dt1 = np.concatenate(([0], np.diff(good_beats)))
        good_dt2 = np.concatenate(([0, 0], good_beats[2:] - good_beats[:-2]))    
        counts3, bins3 = np.histogram(good_dt1, bins=50) # &&ToDo - should this be fixed or auto? 
        counts4, bins4 = np.histogram(good_dt2, bins=50) # just for visual histogram, not used 

        # Refined analysis of beat period and error
        if len(good_beats) > 5:
            # Identify the two most populated bins in the filtered histogram
            top_two_indexes = np.argsort(counts3)[-2:]
            min_idx = np.min(top_two_indexes) # left-most of the two biggest bins
            max_idx = np.max(top_two_indexes) # right-most of the two biggest bins
            
            # Find contiguous non-zero bins around and between these peaks
            while min_idx > 0 and counts3[min_idx - 1] > 0:  # Expand left
                min_idx -= 1
            while max_idx < len(counts3) - 1 and counts3[max_idx + 1] > 0: # Expand right
                max_idx += 1
            
            # Determine the value range from the bin edges
            range_min = bins3[min_idx] # in seconds
            range_max = bins3[max_idx + 1] # in seconds
            
            # Select the beat intervals that fall within this range
            refined_intervals = good_dt1[(good_dt1 >= range_min) & (good_dt1 <= range_max)]
            
            if len(refined_intervals) > 0:
                beat_period = np.mean(refined_intervals)
                
                # Separate into 'ticks' (> avg) and 'tocks' (< avg)
                ticks = refined_intervals[refined_intervals > beat_period]
                tocks = refined_intervals[refined_intervals < beat_period]
                
                tick_interval = np.mean(ticks) if len(ticks) > 0 else beat_period
                tock_interval = np.mean(tocks) if len(tocks) > 0 else beat_period
                
                beat_error_s = abs(tick_interval - tock_interval) / 2
                beat_error_pc = (beat_error_s / beat_period * 100.0) if beat_period > 0 else 0.0

                str = f"Window: {edge_times[0]:.0f}s to {edge_times[-1]:.0f}s. {len(edge_times)} - {len(bad_beats)} = {len(good_beats)} beats. "
                str += f"BPH {3600/crude_beat_period:.1f}s -> {(3600/beat_period):.1f}). "
                str += f"Tick/Tock {tick_interval:.3f}s / {tock_interval:.3f}s -> {beat_error_s:.3f} s ({beat_error_pc:.1f}%)"
                print(str)

                title = f"{clock_name}: {edge_times[0]:.0f}s to {edge_times[-1]:.0f}s - Period {crude_beat_period:.3f}s -> {beat_period:.3f}s"

                if plot_it :
                    windows_plotting.plot_intervals_histogram(counts1, bins1, counts2, bins2, counts3, bins3, counts4, bins4, title)

        return good_beats, bad_beats

        # end weed_edges_in_window() =================================

#==========================  end of class ClockBeatDetector =====================
if __name__ == "__main__":
    print("DON'T RUN THIS - RUN THE FILE THAT CALLS THIS!.")