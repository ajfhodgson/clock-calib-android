import matplotlib.pyplot as plt
import numpy as np
import os

# Windows plotting functions, using matplotlib

def plot_chunk_results(chunk_num, total_chunks, edge_times, debug_info, clock_name='Clock', mad_factor=3):
    """Plot the results for a single chunk."""
    time_axis = debug_info['time_axis']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ax.plot(time_axis, debug_info['audio_chunk'], color='grey', alpha=0.5, label='Audio', linewidth=0.5) # Plot audio in grey
    ax.plot(time_axis, debug_info['filtered'], color='blue', alpha=0.6, label='Filtered', linewidth=0.8) # Plot filtered in blue
    ax.plot(time_axis, debug_info['fast_env'], color='purple', alpha=0.8, label='Fast envelope', linewidth=1)     # Plot fast and slow envelopes in purple
    ax.plot(time_axis, debug_info['slow_env'], color='purple', alpha=0.5, linestyle='--', label='Slow envelope', linewidth=1)
    ax.plot(time_axis, debug_info['onset_strength'], color='green', alpha=0.8, label='Onset strength', linewidth=1.5) # Plot onset strength in green
    ax.axhline(debug_info['threshold'], color='orange', linestyle=':', label='Threshold', linewidth=1) # Plot threshold as horizontal line

    # Plot detected ticks in red
    if len(edge_times) > 0:
        # Find the y-values for the ticks on the onset strength curve
        tick_indices = np.searchsorted(time_axis, edge_times)
        tick_indices = np.clip(tick_indices, 0, len(debug_info['onset_strength']) - 1)
        tick_y_values = debug_info['onset_strength'][tick_indices]
        
        ax.scatter(edge_times, tick_y_values, color='red', s=100, zorder=5, marker='o', label=f'Ticks ({len(edge_times)})')
        
        # Add vertical lines at tick positions
        for tick_time in edge_times:
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


def plot_intervals_histogram(counts1, bins1, counts2, bins2, counts3=None, bins3=None, counts4=None, bins4=None, clock_name='Clock', mad_factor=3):
    """Plot histograms for up to 4 sets of counts and bins in 2x2 subplots."""
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()
    
    data_sets = [
        (counts1, bins1, 'skyblue', 'dt1', 'Beat Interval'),
        (counts2, bins2, 'lightgreen', 'dt2', '2-Beat Interval'),
        (counts3, bins3, 'salmon', 'dt1', 'Weeded Beat Interval'),
        (counts4, bins4, 'wheat', 'dt2', 'Weeded 2-Beat Interval')
    ]
    
    for i, (counts, bins, color, label, title_part) in enumerate(data_sets):
        ax = axs[i]
        if counts is None or bins is None:
            ax.axis('off')
            continue

        bin_widths = np.diff(bins)
        bin_centers = bins[:-1] + bin_widths / 2
        rects = ax.bar(bin_centers, counts, width=bin_widths*0.9, 
                       color=color, edgecolor='black', label=label)
        
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

        bin_span_ms = bin_widths[0] * 1000 if len(bin_widths) > 0 else 0
        total_ticks = np.sum(counts)
        ax.set_title(f'{clock_name} - {title_part}\nBins: {len(bins)-1}, Bin Span: {bin_span_ms:.2f} ms, Total: {total_ticks} ticks, mad_factor: {mad_factor}', fontsize=10)
        ax.set_xlabel('Seconds')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        if len(bins) > 1:
            ax.set_xlim(0, bins[-1])

    plt.tight_layout()
    plt.show()
