import logging
# Suppress debug messages from noisy libraries
logging.getLogger('logger').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('kivy').setLevel(logging.WARNING)
logging.getLogger('deps').setLevel(logging.WARNING)
logging.getLogger('Python').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

import numpy as np
import librosa
import csv
import os

import beat_detector
import windows_plotting
clock_name = 'Long case'
clock_name = 'Djd mantel'
clock_name = 'Ten secs of each'
clock_name = 'Box clock'

clock_name = 'Box Clock 2' # 40s
clock_name = 'DJD Mantel 2' # 40s
clock_name = 'Long Case 2' # 60s
clock_name = 'Box Clock 2' # 40s
clock_name = 'DJD Mantel 2' # 40s


clock_name = 'Djd mantel'


plot_each_chunk = False
plot_histograms = True

beats_analysis_window = 200 # how many edges to accumulate before doing weeding false positives

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


def append_edge_times_to_csv(new_edges, previous_edges):
    """Appends edge times to csv with dt1 and dt2 columns."""
    if len(new_edges) == 0:
        return

    filename = clock_name + "_edges.csv"
    try:
        file_exists = os.path.exists(filename)
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Index","Tick_Time_s", "dt1", "dt2"])
            
            # Combine last 2 previous ticks with new ticks for context
            context = list(previous_edges[-2:]) if len(previous_edges) > 0 else []
            combined = context + list(new_edges) # 2 old plus all of new
            offset = len(context) # typically 2

            for i, tick in enumerate(new_edges):
                idx = i + offset # i is index into new_edges, idx is index into combined
                edge_count = len(previous_edges) + i
                dt1 = combined[idx] - combined[idx-1] if idx >= 1 else ""
                dt2 = combined[idx] - combined[idx-2] if idx >= 2 else ""
                writer.writerow([edge_count, tick, dt1, dt2])
    except Exception as e:
        print(f"I/O Error (append ticks): {e}")


# Usage example with wav file
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
    detector = beat_detector.ClockBeatDetector(sr=44100)
    
    # Replace previous CSV if exists
    if os.path.exists(clock_name +"ticks.csv"):
        os.remove(clock_name +"ticks.csv")

    all_edge_times = [] # entire recording - all chunks
    last_window_start_time = 0 

    for i, chunk in enumerate(chunks):

        chunk_edge_times, debug_info = detector.process_chunk(chunk)
        
        append_edge_times_to_csv(chunk_edge_times, all_edge_times) # appends to ticks file (with dt1 and dt2)
        all_edge_times.extend(chunk_edge_times) # add it to the array itself
        
        if False :
            print(f"Chunk {i+1}/{len(chunks)}: "
                f"{debug_info['time_axis'][0]:.2f}s to {debug_info['time_axis'][-1]:.2f}s - "
                f"found {len(chunk_edge_times)} ticks, total so far: {len(all_edge_times)}"
                )
        if False :
            if len(chunk_edge_times) > 0:
                print(f"Chunk {i} Edge times: {chunk_edge_times}") 
        
        if plot_each_chunk :
            windows_plotting.plot_chunk_results(i+1, len(chunks), chunk_edge_times, debug_info, clock_name=clock_name)

        # if we've accumulated enough edges/potential beats for a weeding session, 
        new_data = len(all_edge_times) - last_window_start_time
        if new_data >= beats_analysis_window:
            edges_to_weed = all_edge_times[-new_data:] # only weed the most recent edges, to see how the histogram evolves over time as more data is added, and to be responsive to changes in tick interval over time 
            good_beats, bad_beats = detector.weed_edges_in_window(edges_to_weed, plot_histograms, clock_name)
            last_window_start_time = len(all_edge_times)

    # end for

    # now analyse/plot the last drips after the beats_analysis_window
    new_data = len(all_edge_times) - last_window_start_time
    edges_to_weed = all_edge_times[-new_data:] # only weed the most recent edges, to see how the histogram evolves over time as more data is added, and to be responsive to changes in tick interval over time 
    good_beats, bad_beats = detector.weed_edges_in_window(edges_to_weed, plot_histograms, clock_name)
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
