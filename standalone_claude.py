import numpy as np
from scipy import signal
import librosa
import csv
import os

import windows_plotting  # keep matplotlib separate

clock_name = 'Long case'
clock_name = 'Djd mantel'
clock_name = 'Ten secs of each'
clock_name = 'Box clock'

clock_name = 'Box Clock 2' # 40s
clock_name = 'DJD Mantel 2' # 40s
clock_name = 'Long Case 2' # 60s


plot_each_chunk = True
plot_histograms = True

mad_factor = 1.0  # Threshold sensitivity factor

beats_analysis_window = 800 # how many ticks to accumulate before plotting histogram of dt1 and dt2 - will get more accurate as more ticks are added, but takes longer to calculate and plot, and less responsive to changes in tick interval over time.    

# Approach to removing false positives: having calculated histograms of dt1 and dt2,
# identify the two highest bins of dt1, take mid-point between them as crude estimate of beat period, 
# and then weed out any beats # where the next beat is a more plausible beat than this one, 
# by comparing how close dt1[i] and dt2[i+1] are to the crude beat period.
# having weeded out false positives, calculate histograms agains, 
# &&ToDo:
# identify all bins which are non-zero and contiguous with the two highest bins. 
# Calculate the average dt1s of all beats falling in these bins to give a more refined beat interval.
# Calculate the average dt1 of ticks from these bins falling above and below the average of these beats
# to give a beat error
# Report how long this calculation and processing takes! Will get progressinvely longer until
# some max window (15 mins?) is reached, then stabilise.

class ClockTickDetector:
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
        # returns their times in seconds (absolute time since ClockTickDetector.reset()) 
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
        self.onset_history.extend(onset_strength.tolist()) # entire history since initialisation of ClockTickDetector class
        if len(self.onset_history) > self.max_history_samples: 
            self.onset_history = self.onset_history[-self.max_history_samples:] # truncate to most recent [10] seconds of data, to adapts to recent conditions
        
        # Calculate adaptive threshold to test onset against, using history - median + mad_factor * MAD 
        history_arr = np.array(self.onset_history)
        valid_history = history_arr[history_arr > 1e-9] # ignore near-zero values to avoid skewing median and MAD
        
        if len(valid_history) > 0:
            median_onset = np.median(valid_history) # halfway between max and minimum of the recent onset strength values
            mad = np.median(np.abs(valid_history - median_onset))
            threshold = median_onset + (mad_factor * mad)
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
        
        debug_info = {
            'time_axis': time_axis,  # Time in seconds for each sample in chunk
            'audio_chunk': audio_chunk,  # Include original audio for plotting
            'onset_strength': onset_strength,
            'threshold': threshold,
            'fast_env': fast_env,
            'slow_env': slow_env,
            'filtered': filtered,
        }
        
        return chunk_edge_times, debug_info

# ======= end of process_chunk method ===========================================

def weed_beats_from_edges(edge_times, plot_it=False, clock_name='Clock', mad_factor=3):
    # called for a decent window's-worth of edges, enough to make a useful histogram
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
    print(f"Crude beat period estimated from histogram: {crude_beat_period:.3f} seconds (bins {idx_low} and {idx_high}, counts {counts1[idx_low]} and {counts1[idx_high]})")

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
            dt2[i+1] = dt2[i+1] + dt1[i] # update dt2 of the next beat to skip the erased edge
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

    print(f"After Weeding: {len(bad_beats)} bad beats, {len(good_beats)} good beats")

    if False :
    # LET'S TRY A SECOND WEEEDING!
        # find mid-point between the two most-populated bins of dt1
        indexes_of_biggest_bins = np.argsort(counts3)[-2:]
        idx_low, idx_high = np.sort(indexes_of_biggest_bins)
        crude_beat_period = (bins3[idx_low] + bins3[idx_high + 1]) / 2
        print(f"Crude beat period estimated from histogram: {crude_beat_period:.3f} seconds (bins {idx_low} and {idx_high}, counts {counts3[idx_low]} and {counts3[idx_high]})")

        beats = good_beats # start optimistically assuming all remaining beats are good, will weed out the false positives using the histograms of dt1 and dt2
        dt1 = good_dt1
        dt2 = good_dt2

        for i in range(2, len(beats)-2): # we've only got dt2 values after i=2 
    #        print(f"Checking beat {i} at time {beats[i]:.2f}s: dt1={dt1[i]:.3f}s, dt2={dt2[i+1]:.3f}s, crude_beat_period={crude_beat_period:.3f}s")
            if (abs(dt1[i] - crude_beat_period) > abs(dt2[i+1] - crude_beat_period)) : # i is BAD - erase it!
                bad_beats.append(beats[i]) # keep track of which beats we are erasing, for reporting and debugging
                dt1[i+1] = dt1[i+1] + dt1[i] # update dt1 of the next beat to skip the erased edge
                dt2[i+1] = dt2[i+1] + dt1[i] # update dt2 of the next beat to skip the erased edge
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

        print(f"After SECOND Weeding: {len(bad_beats)} bad beats, {len(good_beats)} good beats")

    if plot_it :
        windows_plotting.plot_intervals_histogram(counts1, bins1, counts2, bins2, counts3, bins3, counts4, bins4)

    return good_beats, bad_beats

    # end weed_beats_from_edges() =================================

#==========================  end of class ClockTickDetector =====================


def append_edge_times_to_csv(new_ticks, previous_ticks):
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


# Usage example with m4a file
if __name__ == "__main__":

    # Load wav file and resample to 44100 Hz
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

    all_edge_times = [] # entire recording - all chunks
    last_window_start_time = 0 

    for i, chunk in enumerate(chunks):

        chunk_edge_times, debug_info = detector.process_chunk(chunk)
        
        append_edge_times_to_csv(chunk_edge_times, all_edge_times) # appends to ticks file (with dt1 and dt2)
        all_edge_times.extend(chunk_edge_times) # add it to the array itself
        
        print(f"Chunk {i+1}/{len(chunks)}: "
              f"{debug_info['time_axis'][0]:.2f}s to {debug_info['time_axis'][-1]:.2f}s - "
              f"found {len(chunk_edge_times)} ticks, total so far: {len(all_edge_times)}"
              )
#        if len(chunk_edge_times) > 0:
#            print(f"Chunk {i} Edge times: {chunk_edge_times}") 
        
        if plot_each_chunk :
            windows_plotting.plot_chunk_results(i+1, len(chunks), chunk_edge_times, debug_info, clock_name=clock_name, mad_factor=mad_factor)

        # if we've accumulated enough edges/potential beats for a weeding session, 
        new_data = len(all_edge_times) - last_window_start_time
        if new_data >= beats_analysis_window:
            edges_to_weed = all_edge_times[-new_data:] # only weed the most recent edges, to see how the histogram evolves over time as more data is added, and to be responsive to changes in tick interval over time 
            good_beats, bad_beats = weed_beats_from_edges(edges_to_weed, plot_it=plot_histograms, clock_name=clock_name, mad_factor=mad_factor)
            last_window_start_time = len(all_edge_times)

    # end for

    # now analyse/plot the last drips after the beats_analysis_window
    new_data = len(all_edge_times) - last_window_start_time
    edges_to_weed = all_edge_times[-new_data:] # only weed the most recent edges, to see how the histogram evolves over time as more data is added, and to be responsive to changes in tick interval over time 
    good_beats, bad_beats = weed_beats_from_edges(edges_to_weed, plot_it=plot_histograms, clock_name=clock_name, mad_factor=mad_factor)
    last_window_start_time = len(all_edge_times)

    # Calculate (and plot) histograms
    if len(all_edge_times) >= 3:
        ticks = np.array(all_edge_times)
        dt1 = np.diff(ticks)
        dt2 = ticks[2:] - ticks[:-2]
        
        counts1, bins1 = np.histogram(dt1, bins=50)
        counts2, bins2 = np.histogram(dt2, bins=50)

    else:
        print("Not enough ticks for histogram analysis")

    print("\nFine.")
# end main
