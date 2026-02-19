from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy_garden.graph import Graph, MeshLinePlot, ScatterPlot, BarPlot
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
        self.sr = kwargs.pop('sr', 8000) # sample rate
        self.ts_buffer_len = kwargs.pop('ts_buffer_len', int(self.x_span_s * self.sr / 1000)) # buffer length for plotting
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
            font_size='12sp',
            xmin=0, xmax=self.x_span_s,  # initial default this gets rolled on as time goes by
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
            self.ev_buffers[ev_name].extend(ev_data_to_append[ev_name])

            np_time_buffer = np.array(self.ts_buffers['time_axis'])
            if 'onset_strength' in self.ts_buffers:
                np_y_value_buffer = np.array(self.ts_buffers['onset_strength'])
            else:
                np_y_value_buffer = np.zeros(len(self.ts_buffers['time_axis']))

            # Rebuild plot points entirely from the deque each time (never append),
            # so the points list stays bounded to ev_buffer_len and can never overflow.
            # Also filter to only events within the visible x window.
            x_min_visible = self.graph.xmin
            x_max_visible = self.graph.xmax
            new_points = []
            for ev_time in self.ev_buffers[ev_name]:
                if ev_time < x_min_visible or ev_time > x_max_visible:
                    continue
                idx = np.searchsorted(np_time_buffer, ev_time)
                idx = min(idx, len(np_y_value_buffer) - 1)
                    # Get x-value from time series and y-value from onset strength at this time
                    # add to the plotted points
                new_points.append((ev_time, np_y_value_buffer[idx]))
            self.ev_plots[ev_name].points = new_points

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


class HistogramWidget(BoxLayout):
    """
    Kivy widget that draws two histogram bar charts side-by-side using kivy_garden Graph,
    so axis labelling is handled automatically — same as ScrollingGraphWidget.
      Left:  counts1 / bins1  (raw dt1 — all detected edges)
      Right: counts3 / bins3  (weeded dt1 — after false-positive removal)

    Call  update(histogram_data)  with the tuple returned by
    beat_detector.weed_edges_in_window().

    histogram_data = (counts1, bins1, counts2, bins2, counts3, bins3, counts4, bins4, title)
    Only counts1/bins1 and counts3/bins3 are used here.
    X-axis values are in milliseconds.
    """

    def __init__(self, **kwargs):
        super().__init__(orientation='horizontal', **kwargs)

        self._left_graph  = self._make_graph()
        self._right_graph = self._make_graph()

        self._left_plot  = BarPlot(color=[0.3, 0.6, 1.0, 1], bar_spacing=0.9)
        self._right_plot = BarPlot(color=[0.3, 1.0, 0.5, 1], bar_spacing=0.9)

        self._left_graph.add_plot(self._left_plot)
        self._right_graph.add_plot(self._right_plot)

        # bind_to_graph must be called after add_plot so BarPlot can calculate bar widths
        self._left_plot.bind_to_graph(self._left_graph)
        self._right_plot.bind_to_graph(self._right_graph)

        self.add_widget(self._left_graph)
        self.add_widget(self._right_graph)

    def _make_graph(self):
        return Graph(
            x_ticks_minor=0, x_ticks_major=100,
            y_ticks_major=0,            # no y ticks → no reserved y-axis space
            x_grid_label=True, y_grid_label=False,
            x_grid=True, y_grid=False,  # no y gridlines since no y ticks
            padding=5,
            font_size='9sp',            # small enough to fit at compact window size
            xmin=0, xmax=1000,
            ymin=0, ymax=1,
        )

    # ------------------------------------------------------------------
    def update(self, histogram_data):
        """
        histogram_data is the tuple from weed_edges_in_window:
          (counts1, bins1, counts2, bins2, counts3, bins3, counts4, bins4, title)
        """
        counts1, bins1, _c2, _b2, counts3, bins3, _c4, _b4, _title = histogram_data
        self._update_graph(self._left_graph,  self._left_plot,  counts1, bins1)
        self._update_graph(self._right_graph, self._right_plot, counts3, bins3)

    # ------------------------------------------------------------------
    @staticmethod
    def _update_graph(graph, plot, counts, bins):
        """Push new histogram data into one Graph panel. X axis in ms."""
        if counts is None or len(counts) == 0:
            return

        bin_centers_ms = (bins[:-1] + bins[1:]) / 2 * 1000   # ms
        plot.points = list(zip(bin_centers_ms, counts))

        # Update axis ranges
        x_max_ms = bins[-1] * 1000
        y_max    = max(counts) if max(counts) > 0 else 1

        graph.xmin = 0
        graph.xmax = float(x_max_ms)
        graph.ymin = 0
        graph.ymax = float(y_max) * 1.1

        # Aim for ~5 major x ticks
        tick = max(1, round(x_max_ms / 5 / 50) * 50)   # round to nearest 50 ms
        graph.x_ticks_major = float(tick)


#==========================  end of class HistogramWidget() ======================
if __name__ == "__main__":
    print("DON'T RUN THIS - RUN THE FILE THAT CALLS THIS!.")