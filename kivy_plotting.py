from kivy.uix.boxlayout import BoxLayout
from kivy_garden.graph import Graph, MeshLinePlot, ScatterPlot
from collections import deque
import numpy as np

'''
Sets up the Kivy widget for plotting:
time series data (arrays at fixed sample rate) and 
event data (array values are timestamps in seconds)

This is a custom widget that contains a Graph and manages the data buffers for plotting, so the 
calling program can just add more data and this widget takes care of holding historical data (in deques), 
adding new data to the deques, and plotting a window's worth of time series and event data, giving a jumpy scrolling effect

The main class is ScrollingGraphWidget, which has methods to add new chunk data to the plot and to clear the buffers 
when starting a new run.

The time series data (audio/filtered/onset etc) data passed to add_chunk_to_plot() is already downsampled (e.g. by 1000),
to the resolution needed to be plotted. In here we just plot what we're given.

Note that the adding new event data has to cope with the new data potentially having overlap in time with the
existing historical data

'''

class ScrollingGraphWidget(BoxLayout):
    def __init__(self, **kwargs):
        # can take custom arguments 
        # x_span_s              for span of x axis in seconds
        # ts_buffer_len         how big timeseries buffers need to be
        # ev_buffer_len         how big event buffers need to be
        # chart_colours         colours to be used for each named trace

        #&&ToDo get rid of these once calling program reliably passes them in, to avoid maintenance headache
        # time_series_data is {'time_axis', 'audio_chunk', 'onset_strength', 'threshold', 'fast_env', 'slow_env', 'filtered'}

        default_colours = {             # set them all up, in case of need later
                'audio_chunk':  [0.5, 0.5, 0.5, 0.5],   # grey
                'filtered':     [0, 0, 1, 0.6],         # Blue
                'onset_strength':        [1, 1, 1, 0.8],         # White
                'threshold':    [1, 0, 1, 0.8],         # Magenta
                'fast_env':     [0.5, 0, 0.5, 0.8],     # Purple
                'slow_env':     [0.5, 0, 0.5, 0.5],     # Purple faded

                'edges':        [1, 0, 1, 0.8],     # Magenta
                'ticks':        [0, 1, 0, 0.6],     # Green
                'noises':       [1, 0, 0, 0.8],     # Red
        }

        #&&ToDo - put in checks for any zero-length datasets passed in
        #&&ToDo - bugger - I forgot!

        # Extract custom arguments before calling super() to avoid Kivy unknown arg error
        self.x_span_s = kwargs.pop('x_span_s', 60) # seconds to show on x-axis
        self.ts_buffer_len = kwargs.pop('ts_buffer_len', int(self.x_span_s * 44100 / 1000)) # buffer length for plotting
        self.ev_buffer_len = kwargs.pop('ev_buffer_len', int(self.x_span_s * 20)) # buffer length for plotting events

        self.colours = kwargs.pop('chart_colours', default_colours) 
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        
        # Track absolute time
        self.latest_time = 0.0

        self.ts_buffers = {} # will contain named buffers, one for each timeseries asked to be plotted
        self.ev_buffers = {} # will contain named buffers, one for each event series asked to be plotted
        self.ts_plots = {}    # container for kivy_graphics plots
        self.ev_plots = {}    # container for kivy_graphics plots

        # Create graph itself on which we draw lines and points (no plots on it yet!)
        self.graph = Graph(
            x_ticks_minor=5, x_ticks_major=10, y_ticks_major=0.001, 
            x_grid=True, y_grid=True, y_grid_label=True, x_grid_label=True, padding=5,
            xmin=0, xmax=self.x_span_s, # initial default this gets rolled on as time goes by
            ymin=-0.01, ymax=0.0001 # initial default - we'll auto-scale this based on data in the current window
        )
                
        self.add_widget(self.graph)
        
#-------------------------------------------------------------------------------------------------------    
    def add_chunk_to_plot(self, ts_data_to_append, ev_data_to_append):
        # NOTE - the timeseries data passed in here is already downsampled and in numpy arrays
        # NOTE - the event data passed in here may overlap data we've already seen
        # ts_data+to_append['time_series'] is a special, 'known' timeseries name

        # Update current time (end of the passed chunk data)
        self.latest_time = ts_data_to_append['time_axis'][-1]

        # Append new chunk of downsampled timeseries data to deque buffers
        for ts_name in ts_data_to_append:
            # create the buffers and the plot, if it's new
            if ts_name not in self.ts_buffers: # not seen this before! Create the buffer and line plot
                self.ts_buffers[ts_name] = deque(maxlen=self.ts_buffer_len)
                if ts_name != 'time_axis': # no plot for time itself!
                    self.ts_plots[ts_name] = MeshLinePlot(color=self.colours[ts_name])
                    self.graph.add_plot(self.ts_plots[ts_name])
            # append the new data to the history deque
            self.ts_buffers[ts_name].extend(ts_data_to_append[ts_name]) # simply append he new data to the correct buffer
            # add the new points to the plot (x = time, y = ts_name)
            if ts_name != 'time_axis': # update the actual plot points in the plot
                self.ts_plots[ts_name].points = [(t, f) for t, f in zip(self.ts_buffers['time_axis'], self.ts_buffers[ts_name])]

        for ev_name in ev_data_to_append:
            if ev_name not in self.ev_buffers: # not seen this before! Create the buffer and scatter plot
                self.ev_buffers[ev_name] = deque(maxlen=self.ev_buffer_len)
                self.ev_plots[ev_name] = ScatterPlot(color=self.colours[ev_name], point_size=3)
                self.graph.add_plot(self.ev_plots[ev_name])
            self.ev_buffers[ev_name].extend(ev_data_to_append[ev_name]) # simply append he new data to the correct buffer

            #&&ToDo - take care of the potential overlap - only add new data points later than the existing last event time
            # for events, we just have times in s. What to use for y-value? Could be 0 for simplicity for now
            # otherwise we have to look through ts_buffers['time_axis'] for a matching time, then look this time up in
            # one of the other ts_series, e.g. 'filtered' or 'onset_strength', which would require knowing which in here 
            # (I'd prefer to be open_minded)

            np_time_buffer = np.array(self.ts_buffers['time_axis']) # outside the loop - need it for all event series
            #&&ToDo - should pass in the name of the timeseries whose y-value to plot evets at, not guess the name!
            if 'onset_strength' in self.ts_buffers:
                np_y_value_buffer = np.array(self.ts_buffers['onset_strength'])
            else:
                np_y_value_buffer = np.zeros(len(self.ts_buffers['time_axis']))

            if len(self.ev_buffers[ev_name]) == 0:
                self.ev_plots[ev_name].points = []
            else:
                # For each edge time, find the corresponding y-value on the onset data
                for ev_time in self.ev_buffers[ev_name]:
                # Find the closest time in the timescale array
                    idx = np.searchsorted(np_time_buffer, ev_time)
                    idx = min(idx, len(np_y_value_buffer) - 1)
                    # Get x-value from time series and y-value from onset strength at this time
                    # add to the plotted points
                    self.ev_plots[ev_name].points.append((ev_time, np_y_value_buffer[idx]))

        # Update x-axis to show scrolling window - show the most recent x_span_s
        if self.latest_time <= self.x_span_s:   # Still filling up the first window
            self.graph.xmin = 0 
            self.graph.xmax = self.x_span_s
        else: # Scroll the window
            self.graph.xmin = float(int(self.latest_time - self.x_span_s)) # can't cope with an np_float result
            self.graph.xmax = float(int(self.graph.xmin + self.x_span_s)) # can't cope with an np_float result
            
            # Force a tick update to ensure labels don't disappear
            self.graph.x_ticks_major = self.graph.x_ticks_major
        
        # Auto-scale y-axis based on plotted events
        if len(np_y_value_buffer) > 0:
            self.graph.ymin = 0
            self.graph.ymax = max(0.0001, float(np.max(np_y_value_buffer)) * 1.1) # add 10% headroom
      
    
    def clear_buffers(self):
        """Clear all buffers and reset graph. at the start of a new run"""
        for ts_name in self.ts_buffers:
            self.ts_buffers[ts_name].clear()
        for ts_name in self.ts_plots:
            self.ts_plots[ts_name].points = []

        for ev_name in self.ev_buffers:
            self.ev_buffers[ev_name].clear()
        for ev_name in self.ev_plots:
            self.ev_plots[ev_name].points = []

        self.latest_time = 0.0
        self.graph.xmin = 0
        self.graph.xmax = self.x_span_s
        
#==========================  end of class ScrollingGraphWidget() =====================
if __name__ == "__main__":
    print("DON'T RUN THIS - RUN THE FILE THAT CALLS THIS!.")