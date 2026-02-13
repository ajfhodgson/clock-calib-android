from kivy.uix.boxlayout import BoxLayout
from kivy_garden.graph import Graph, MeshLinePlot, ScatterPlot
from collections import deque
import numpy as np

'''
Sets up the Kivy widget for plotting the audio data and detected edges.
This is a custom widget that contains a Graph and manages the data buffers for plotting the audio, onset strength, and detected edges in real-time as the audio is processed in chunks.
The main class is ScrollingGraphWidget, which has methods to add new chunk data to the plot and to clear the buffers when starting a new run.

The timebase/audio/filtered/onset etc data passed to add_chunk_to_plot() is already downsampled (e.g. by 1000)
And needs appending to deque buffers
BUT edge times are not downsampled, they are absolute seconds snce start of listening
At the end of most chunks we just have edge times for this chunk.
But if the chunk end is a window_weed time, we have good and bad beat times going further back, so it's not a 
matter of simply appending the new edge times to the buffer, we need to replace the buffer with the most recent good and bad beats (and later, ticks and tocks) for the most recent window of time. So we will need to call a method like update_edge_times(good_beats, bad_beats) that replaces the edge times buffer with the good and bad beats (and later ticks and tocks) from the most recent window. This way the scatter plot will show only the categorised edges (good beats, bad beats, ticks, tocks) rather than all detected edges.
I'll probably handle this by letting edge times be simply appended to the deque, but having passed data for
good, bad, ticks, tocks spannning the whole plotted period, wthey will overwrite the edges already plotted
(though I could remove any edges that coincide with goods, bads, ticks, tocks to avoid any chance of double plotting - but maybe it's interesting to see the original edges that got categorised as good beats, bad beats, ticks, tocks for efficiency)

'''

class ScrollingGraphWidget(BoxLayout):
    def __init__(self, **kwargs):

        # Extract custom arguments before calling super() to avoid Kivy unknown arg error
        self.x_span_s = kwargs.pop('x_span_s', 60) # seconds to show on x-axis
        self.ds_buffer_len = kwargs.pop('ds_buffer_len', self.x_span_s * 44100 // 1000) # buffer length for plotting
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        
        # Rolling buffer parameters

       
        # Data buffers - hold x_span_s seconds of downsampled data
        self.buffer_size = self.ds_buffer_len  

        self.time_buffer = deque(maxlen=self.buffer_size)
        self.filtered_buffer = deque(maxlen=self.buffer_size)
        self.onset_buffer = deque(maxlen=self.buffer_size)
        self.threshold_buffer = deque(maxlen=self.buffer_size)
#        self.audio_buffer = deque(maxlen=self.buffer_size)
#        self.fast_env_buffer = deque(maxlen=self.buffer_size)
#        self.slow_env_buffer = deque(maxlen=self.buffer_size)

        # scatter plot (edges, beats, ticks, tocks) data storage
        self.edge_times_buffer = deque(maxlen=100)  # Store edge times

        # Track absolute time
        self.current_time = 0.0
        
        # Create graph
        self.graph = Graph(
#-            xlabel='Time (s)', ylabel='Amplitude',
            x_ticks_minor=5, x_ticks_major=10, y_ticks_major=0.001, 
            x_grid=True, y_grid=True, y_grid_label=True, x_grid_label=True, padding=5,
            xmin=0, xmax=self.x_span_s, # initial default this gets rolled on as time goes by
            ymin=-0.01, ymax=0.01 # initial default - we'll auto-scale this based on data in the current window
        )
        
        # Create plots - each line on the chart is considered a "plot" in kivy_garden.graph
        self.filtered_plot = MeshLinePlot(color=[0, 0, 1, 0.6])     # Blue
        self.onset_plot = MeshLinePlot(color=[0, 1, 0, 0.8])        # Green
        self.threshold_plot = MeshLinePlot(color=[1, 1, 0, 0.8])        # Magenta
#        self.audio_plot = MeshLinePlot(color=[0.5, 0.5, 0.5, 0.5])  # Grey
#        self.fast_env_plot = MeshLinePlot(color=[0.5, 0, 0.5, 0.8]) # Purple
#        self.slow_env_plot = MeshLinePlot(color=[0.5, 0, 0.5, 0.5]) # Purple faded
        
        # Create scatter plot for edge markers (red dots) - in future will be 
        # good beats and bad beats, maybe ticks and tocks in different colours
        # uncategorsised edges will show before the WindowWeeder weeds them and categorises them as good beats or bad beats
#&&ToDo        self.edge_plot = ScatterPlot(color=[1, 0, 0, 1], pointsize=5)  # Red
        self.edge_plot = ScatterPlot(color=[1, 0, 0, 1], point_size=3)  # Red
        
        # Add plots to graph
        self.graph.add_plot(self.filtered_plot)
        self.graph.add_plot(self.onset_plot)
        self.graph.add_plot(self.threshold_plot)
#        self.graph.add_plot(self.audio_plot)
#        self.graph.add_plot(self.fast_env_plot)
#        self.graph.add_plot(self.slow_env_plot)
        self.graph.add_plot(self.edge_plot)  # Add edge markers on top
        
        self.add_widget(self.graph)
        
#-------------------------------------------------------------------------------------------------------    
    def add_chunk_to_plot(self, downsampled_data, edge_times):
        # NOTE - the data passed in here is already downsampled and in numpy arrays
        # Update current time (end of passed chunk data)
        self.current_time = downsampled_data['time_axis'][-1]
        # Append new chunk downsampled data to deque buffers
        self.time_buffer.extend(downsampled_data['time_axis'])
        self.filtered_buffer.extend(downsampled_data['filtered'])
        self.onset_buffer.extend(downsampled_data['onset_strength'])
        self.threshold_buffer.extend(downsampled_data['threshold'])
#        self.audio_buffer.extend(downsampled_data['audio_chunk'])
#        self.fast_env_buffer.extend(downsampled_data['fast_env'])
#        self.slow_env_buffer.extend(downsampled_data['slow_env'])
        self.edge_times_buffer.extend(edge_times) # holds more than we need, but gets clipped in plotting
        
        # Update the graph
        if len(self.time_buffer) == 0:
            return
        
        # Update plot data
#        self.audio_plot.points =    [(t, a) for t, a in zip(self.time_buffer, self.audio_buffer)]
        self.filtered_plot.points = [(t, f) for t, f in zip(self.time_buffer, self.filtered_buffer)]
#        self.fast_env_plot.points = [(t, f) for t, f in zip(self.time_buffer, self.fast_env_buffer)]
#        self.slow_env_plot.points = [(t, s) for t, s in zip(self.time_buffer, self.slow_env_buffer)]
        self.onset_plot.points = [(t, o) for t, o in zip(self.time_buffer, self.onset_buffer)]
        self.threshold_plot.points = [(t, o) for t, o in zip(self.time_buffer, self.threshold_buffer)]
        
        # Update x-axis to show scrolling window
        # Show the most recent x_span_s
        if self.current_time <= self.x_span_s:             # Still filling up the first window
            self.graph.xmin = 0 
            self.graph.xmax = self.x_span_s
        else: # Scroll the window
            self.graph.xmin = float(int(self.current_time - self.x_span_s)) # can't cope with an np_float result
            self.graph.xmax = float(int(self.graph.xmin + self.x_span_s)) # can't cope with an np_float result
            
            # Force a tick update to ensure labels don't disappear
            self.graph.x_ticks_major = self.graph.x_ticks_major
        
        # Auto-scale y-axis based on visible data
        if len(self.onset_buffer) > 0:
            self.graph.ymin = 0
            self.graph.ymax = float(np.max(self.onset_buffer)) * 1.1 # add 10% headroom
        
        # Update edge markers
        if len(self.edge_times_buffer) == 0:
            self.edge_plot.points = []
        else:        
            # For each edge time, find the corresponding y-value on the onset data
            np_time_buffer = np.array(self.time_buffer)
            np_onset_buffer = np.array(self.onset_buffer)
            for edge_time in self.edge_times_buffer:
                # Find the closest time in the timescale array
                idx = np.searchsorted(np_time_buffer, edge_time)
                idx = min(idx, len(np_onset_buffer) - 1)
                    
                # Get y-value from onset strength at this time and add to the plotted points
                y_value = self.onset_buffer[idx]
                self.edge_plot.points.append((edge_time, y_value))
        
    
    def clear_buffers(self):
        """Clear all buffers and reset graph. at the start of a new run"""
        self.time_buffer.clear()
        self.filtered_buffer.clear()
        self.onset_buffer.clear()
        self.threshold_buffer.clear()
#        self.audio_buffer.clear()
#        self.fast_env_buffer.clear()
#        self.slow_env_buffer.clear()
        self.edge_times_buffer.clear()

        self.filtered_plot.points = []
        self.onset_plot.points = []
        self.threshold_plot.points = []
#        self.audio_plot.points = []
#        self.fast_env_plot.points = []
#        self.slow_env_plot.points = []
        self.edge_plot.points = []
        self.current_time = 0.0
        
        self.graph.xmin = 0
        self.graph.xmax = self.x_span_s
        
#==========================  end of class ScrollingGraphWidget() =====================
if __name__ == "__main__":
    print("DON'T RUN THIS - RUN THE FILE THAT CALLS THIS!.")