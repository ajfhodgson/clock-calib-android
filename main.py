import numpy as np
from kivy.utils import platform
import os

# Set KIVY_HOME on Android to a writable directory before other Kivy imports.
# This is to fix a "Permission denied" error when Kivy tries to copy
# its icon files to a non-writable location on startup.
if platform == 'android':
    from jnius import autoclass
    PythonActivity = autoclass('org.kivy.android.PythonActivity')
    activity = PythonActivity.mActivity
    os.environ['KIVY_HOME'] = activity.getFilesDir().getAbsolutePath()

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView
from kivy.graphics import Color, Rectangle
from kivy.clock import Clock
from kivy.utils import platform
from kivy.core.window import Window
import threading
import csv
import os
from datetime import datetime
from collections import deque

# my code in other files:
import beat_detector # clock tick detector (lots of help from Claude AI)
import kivy_plotting # custom Kivy widget for plotting scrolling graphs of audio and detected beats


# Mask bits for tell() messages:
# Bit 0 (1): Startup and status/error messages
# Bit 1 (2): 
# Bit 2 (4): File writing messages
# Bit 3 (8): Chunk Analysis summary
# Bit 4 (16): Chunk Analysis detail 
# Bit 5 (32): WindowWeeder summary
# 

#&&ToDo - avoid the crash on Android after first run/request for microphone permission
#&&ToDo - ensure there is a timestamp (ms since pressing Start) with the amplitude data in CSV
#&&ToDo - Add a scrolling plot of the received amplitudes, marking the detected peaks


# --- HELPER FUNCTIONS FOR FILE I/O ---
def get_save_directory():
    """Get the appropriate save directory based on platform."""
    if platform == 'android':
        # Use jnius to get the Downloads directory on Android
        try:
            from jnius import autoclass
            Environment = autoclass('android.os.Environment')
            Downloads = Environment.getExternalStoragePublicDirectory(
                Environment.DIRECTORY_DOWNLOADS
            )
            base_dir = Downloads.getAbsolutePath()
        except Exception:
            # Fallback to hardcoded path if jnius fails
            base_dir = '/sdcard/Download'
    else:
        # Windows/other: use standard roaming APPDATA folder
        base_dir = os.path.join(os.environ.get('APPDATA', App.get_running_app().user_data_dir), 'clock-calib')
    
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

# --- TIER 2: ANALYSIS & I/O WORKER ---
def write_csv(app, filename_root, timestamp, headers, columns):
    """
    ***This runs in its own thread!***
    Generic function to write columns of data to a CSV file.
    columns: list of numpy arrays (must be same length)
    """
    filename = f"{timestamp}_{filename_root}.csv"
    try:
        base_dir = get_save_directory()
        filepath = os.path.join(base_dir, filename)

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(zip(*columns))

        # verify file has been written and where 
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            app.tell(f"[Tier 2] {filename} written ({file_size} bytes)",2)
        else:
            app.tell(f"[Tier 2] ERROR: File not created - {filepath}",2)

    except Exception as e:
        app.tell(f"[Tier 2] I/O Error ({filename_root}): {e}")

#############################################-----------------------------------------------

class ClockApp(App):
    def build(self): # required by kivy, builds the UI and initializes state
        if platform == 'win': # make Windows look like the Android Samsung S10
            Window.size = (352, 750)

        # Configuration
        # UI-changeable defaults
        self.target_bph = 8860 # sriaght-sided mantel clock - default - can be set in UI
        self.tell_mask = 1 + 2 + 8 + 16   # Controls which tell() messages are emitted; bitmask flags (default: just bit 0 = 1)

        # set parameters
        self.sample_rate = 8000
        self.chunk_time_s = 4.0 
        self.chunks_per_window = 4 # likely to want 15 in prod
        self.audio_chart_time_s = 60.0
        self.audio_downsample_factor = 1000
        self.beats_chart_time_s = 120.0
        self.peak_thresh_pc = 0.1 # not currently used

        self.window_time_s = self.chunk_time_s * self.chunks_per_window # 60 seconds - how long the window should be for weeding false positives. Shorter is more responsive to changes in tick interval, but less data for histogram analysis. Longer is less responsive to changes in tick interval, but more data for histogram analysis.
        self.samples_per_chunk = int(self.sample_rate * self.chunk_time_s)

        # State
        self.is_running = False
        self.stream = None
        self.ms_since_start = 0.0  # running ms counter (dead-reckoning since Start pressed)


        # Double-buffer (swing buffer) for lock-free operation transferring auido chunks from Tier 1 (audio callback) to Tier 2 (analysis and file writing).
        self.buffer_a = []
        self.buffer_b = []
        self.active_buffer = self.buffer_a  # Tier 1 appends to this
        self.inactive_buffer = self.buffer_b  # Tier 2 processes this

        # deques for accumulating data while running
        self.edge_times_deque = deque(maxlen=4000) # for accumulating edge times for histogram analysis and weeding false positives
        self.ticks_deque = deque(maxlen=4000)
        self.noises_deque = deque(maxlen=4000)
        self.chunk_counter = 0


        # Initialize Claude's clock tick detector
        self.detector = beat_detector.ClockBeatDetector(sr=self.sample_rate, min_tick_interval=0.3)

        if True :    # Initialise UI 

            layout = BoxLayout(orientation='vertical', padding=30, spacing=20) # complete screen
            
            # Helper to scale font size based on widget height
            def autoscale(widget, factor=0.5):
                widget.bind(height=lambda w, h: setattr(w, 'font_size', h * factor))

            # Buttons layout - horizontal at top (10% height)  -------------------------------------------------
            buttons_layout = BoxLayout(orientation='horizontal', size_hint_y=0.10, spacing=10)
            
            self.start_btn = Button(text="Start", size_hint_x=1)
            self.start_btn.bind(on_press=self.start_session)
            autoscale(self.start_btn, 0.9)
            buttons_layout.add_widget(self.start_btn)

            self.stop_btn = Button(text="Stop", disabled=True, size_hint_x=1)
            self.stop_btn.bind(on_press=self.stop_session)
            autoscale(self.stop_btn, 0.9)
            buttons_layout.add_widget(self.stop_btn)

            self.exit_btn = Button(text="Exit", size_hint_x=1)
            self.exit_btn.bind(on_press=self.exit_app)
            autoscale(self.exit_btn, 0.9)
            buttons_layout.add_widget(self.exit_btn)
            
            layout.add_widget(buttons_layout)

            # Entry fields row - horizontal (10% height) -------------------------------------------------
            entries_layout = BoxLayout(orientation='horizontal', size_hint_y=0.10, spacing=10)

            # chunk Size (seconds)
            ws_box = BoxLayout(orientation='horizontal')
            l1 = Label(text='Chunk\n(sec)', size_hint_x=1.0)
            autoscale(l1, 0.3)
            ws_box.add_widget(l1)

            self.win_input = TextInput(text=str(self.chunk_time_s), multiline=False, input_filter='float', size_hint_x=0.9)
            autoscale(self.win_input, 0.4)
            self.win_input.bind(on_text_validate=self.on_window_input, on_focus=self.on_window_focus)
            ws_box.add_widget(self.win_input)
            entries_layout.add_widget(ws_box)

            # Target BPH
            tb_box = BoxLayout(orientation='horizontal')
            l2 = Label(text='Tgt\nBPH', size_hint_x=1.0)
            autoscale(l2, 0.3)
            tb_box.add_widget(l2)
            self.bph_input = TextInput(text=str(getattr(self, 'target_bph', 0)), multiline=False, input_filter='int', size_hint_x=0.9)
            autoscale(self.bph_input, 0.4)
            self.bph_input.bind(on_text_validate=self.on_bph_input, on_focus=self.on_bph_focus)
            tb_box.add_widget(self.bph_input)
            entries_layout.add_widget(tb_box)

            # Peak Threshold %
            pt_box = BoxLayout(orientation='horizontal')
            l3 = Label(text='Peaks\n(%)', size_hint_x=1.0)
            autoscale(l3, 0.3)
            pt_box.add_widget(l3)
            self.peak_input = TextInput(text=str(getattr(self, 'peak_thresh_pc', 0.1)), multiline=False, input_filter='float', size_hint_x=0.9)
            autoscale(self.peak_input, 0.4)
            self.peak_input.bind(on_text_validate=self.on_peak_input, on_focus=self.on_peak_focus)
            pt_box.add_widget(self.peak_input)
            entries_layout.add_widget(pt_box)

            # Tell mask (bitmask input)
            ts_box = BoxLayout(orientation='horizontal')
            l4 = Label(text='Tell\nmask', size_hint_x=1.0)
            autoscale(l4, 0.3)
            ts_box.add_widget(l4)
            self.tell_input = TextInput(text=str(getattr(self, 'tell_mask', 1)), multiline=False, input_filter='int', size_hint_x=0.9)
            autoscale(self.tell_input, 0.4)
            self.tell_input.bind(on_text_validate=self.on_tell_input, on_focus=self.on_tell_focus)
            ts_box.add_widget(self.tell_input)
            entries_layout.add_widget(ts_box)

            layout.add_widget(entries_layout)

            # Enable inputs (start disabled when recording)
            self.set_inputs_enabled(True)

            # Status label (10% height)
            self.status_label = Label(
                text="Ready to Record. Press Start.", 
                halign='center', font_size='12sp',
                size_hint_y=0.10
            )
            layout.add_widget(self.status_label)

            # Scrolling text region (30% of screen)
            self.log_scroll = ScrollView(
                size_hint_y=0.3,
                do_scroll_x=False,
                do_scroll_y=True,
                scroll_type=['bars', 'content'],
                bar_width=10
            )
            with self.log_scroll.canvas.before:
                Color(0.2, 0.2, 0.2, 1)
                self.log_rect = Rectangle(size=self.log_scroll.size, pos=self.log_scroll.pos)
            self.log_scroll.bind(pos=self._update_scroll_rect, size=self._update_scroll_rect)

            self.log_label = Label(text='', font_size='12sp', size_hint_y=None, color=(1, 1, 1, 1), halign='left', valign='top')
            self.log_label.bind(texture_size=lambda instance, value: setattr(instance, 'height', value[1]))
            self.log_label.bind(width=lambda instance, value: setattr(instance, 'text_size', (value, None)))
            
            self.log_scroll.add_widget(self.log_label)
            layout.add_widget(self.log_scroll)

            # Bottom section with 3 equal areas
            bottom_layout = BoxLayout(orientation='vertical', size_hint_y=1, spacing=5)
            
#            # Area 1: Stats
#            self.stats_area = BoxLayout(orientation='vertical')
#            self.stats_area.add_widget(Label(text='Stats Area'))
#            bottom_layout.add_widget(self.stats_area)
            
            # Area 2: Timegrapher Chart
            self.timegrapher_chart = BoxLayout(orientation='vertical')
            with self.timegrapher_chart.canvas.before:
                Color(0, 0.1, 0, 1)
                self.tg_rect = Rectangle(size=self.timegrapher_chart.size, pos=self.timegrapher_chart.pos)
            self.timegrapher_chart.bind(size=lambda instance, value: setattr(self.tg_rect, 'size', value))
            self.timegrapher_chart.bind(pos=lambda instance, value: setattr(self.tg_rect, 'pos', value))
            self.timegrapher_chart.add_widget(Label(text='Timegrapher Chart'))
            bottom_layout.add_widget(self.timegrapher_chart)
            
            # Area 3: Audio Chart
            self.audio_chart = kivy_plotting.ScrollingGraphWidget(x_span_s=self.audio_chart_time_s, sr=self.sample_rate)
            bottom_layout.add_widget(self.audio_chart)

            layout.add_widget(bottom_layout)

            self.tell(f"[Init] Screen Size: {Window.size}")

        # Attempt to delete old CSVs on startup
        deleted = 0
        failed = 0
        try:
            save_dir = get_save_directory()
            self.tell(f"[Init] CSVs folder: {save_dir}", 0)
            for filename in os.listdir(save_dir):
                if filename.lower().endswith(".csv"):
                    try:
                        os.remove(os.path.join(save_dir, filename))
                        deleted += 1
                    except Exception as e:
                        failed += 1
        except Exception as e:
            self.tell(f"[Init] Error accessing save directory: {e}")
        self.tell(f"[Init] {deleted} CSVs deleted, {failed} failed to be deleted", 0)

        return layout

    def _update_scroll_rect(self, instance, value):
        self.log_rect.pos = instance.pos
        self.log_rect.size = instance.size

    # --- TIER 1: HIGH-PRIORITY LISTENER --- appends to the current buffer
    def audio_callback(self, indata, frames, time_info, status):
        """Hardware-level callback. Minimal logic to avoid underruns."""
        if self.is_running:
            # Append to whichever buffer is currently active.
            # No lock needed: Tier 1 only appends to active_buffer,
            # and Tier 2 only reads from inactive_buffer.
            self.active_buffer.extend(indata.flatten().tolist())
#--            print(f"[Tier 1] Audio callback: appended {samples_count} samples (active buffer now has {len(self.active_buffer)} total)")

    # --- TIER 3: UI THREAD CONTROLS ---
    def tell(self, message, mask_bit=0):
        """Write a message to the scrolling text region and optionally to console.
        Only outputs if bit mask_bit is set in `self.tell_mask`.
        """
        # If the requested mask_bit is not set in tell_mask, skip output
        try:
            if not ((1 << mask_bit) & getattr(self, 'tell_mask', 1)):
                return
        except Exception:
            pass
        mess = datetime.now().strftime("%H:%M:%S.%f")[:-5] + ' ' + str(mask_bit) + ': ' + message
        if platform == 'win':
            print(mess)
        # Schedule the UI update to the scrolling message window on the main thread
        Clock.schedule_once(lambda dt: self._update_log( mess ), 0)
    
    def _update_log(self, message): 
        """Internal method to update the log text (runs on main thread)."""
        # Add to text region with newline
        if self.log_label.text:
            self.log_label.text += '\n' + message
        else:
            self.log_label.text = message
        
        # Scroll to bottom (scroll_y=0 is bottom in Kivy)
        self.log_scroll.scroll_y = 0

    def exit_app(self, instance):
        """Exit the application."""
        if self.is_running:
            self.stop_session(None)
        self.stop()

    # --- UI input handlers ---
    def _parse_float(self, text, default):
        try:
            return float(text)
        except Exception:
            return default

    def _parse_int(self, text, default):
        try:
            return int(float(text))
        except Exception:
            return default

    def _set_chunk_time_s(self, text):
        val = self._parse_float(text, self.chunk_time_s)
        if val <= 0:
            self.tell("[UI] Invalid chunk duration, keeping previous value")
            self.win_input.text = str(self.chunk_time_s)
            return
        self.chunk_time_s = val
        self.samples_per_chunk = int(self.sample_rate * self.chunk_time_s)
        self.tell(f"[UI] chunk duration set to {self.chunk_time_s}s")
        if self.is_running:
            Clock.unschedule(self.process_chunk_buffer)
            Clock.schedule_interval(self.process_chunk_buffer, self.chunk_time_s)
            self.tell("[UI] Rescheduled buffer processing to new chunk duration")

    def on_window_input(self, instance):
        self._set_chunk_time_s(instance.text)

    def on_window_focus(self, instance, value):
        if not value:
            self._set_chunk_time_s(instance.text)

    def on_bph_input(self, instance):
        val = self._parse_int(instance.text, self.target_bph)
        self.target_bph = val
        self.tell(f"[UI] Target BPH set to {self.target_bph}")

    def on_bph_focus(self, instance, value):
        if not value:
            self.on_bph_input(instance)

    def on_peak_input(self, instance):
        val = self._parse_float(instance.text, self.peak_thresh_pc)
        self.peak_thresh_pc = val
        self.tell(f"[UI] Peak threshold set to {self.peak_thresh_pc}%")

    def on_peak_focus(self, instance, value):
        if not value:
            self.on_peak_input(instance)

    def on_tell_input(self, instance):
        val = self._parse_int(instance.text, getattr(self, 'tell_mask', 1))
        if val < 0:
            self.tell("[UI] Invalid tell mask value; must be non-negative")
            self.tell_input.text = str(self.tell_mask)
            return
        self.tell_mask = val
        self.tell(f"[UI] Tell mask set to {self.tell_mask}")

    def on_tell_focus(self, instance, value):
        if not value:
            self.on_tell_input(instance)

    def set_inputs_enabled(self, enabled: bool):
        """Enable/disable the three entry inputs and adjust appearance."""
        # TextInput has 'disabled', 'background_color', and 'foreground_color'
        bg_enabled = (1, 1, 1, 1)
        bg_disabled = (0.9, 0.9, 0.9, 1)
        fg_enabled = (0, 0, 0, 1)
        fg_disabled = (0.5, 0.5, 0.5, 1)

        for widget in (self.win_input, self.bph_input, self.peak_input, self.tell_input):
            widget.disabled = not enabled
            widget.background_color = bg_enabled if enabled else bg_disabled
            widget.foreground_color = fg_enabled if enabled else fg_disabled

    def start_android_audio(self):
        try:
            from android_audio import AndroidMic
            self.mic = AndroidMic(sample_rate=self.sample_rate)
            self.tell("[Android] AndroidMic initialized, starting callback loop...")
            self.mic.start(callback=self.audio_callback)
            self.tell("[Android] AndroidMic started")
            
            Clock.schedule_interval(self.process_chunk_buffer, self.chunk_time_s)
            self.tell("[Main] Buffer processing scheduled")
        except Exception as e:
            self.tell(f"CRITICAL ERROR: {e}")
            self.status_label.text = f"Mic Error: {e}"
            self.stop_session(None)

    def permission_callback(self, permissions, results):
        if all(results):
            self.tell(f"[Android] {results} Permissions granted.")
            self.start_android_audio()
        else:
            self.tell("[Android] {results} Permissions denied.")
            self.status_label.text = "Permissions Denied"
            self.stop_session(None)

    def start_session(self, instance):
        self.tell("Start button clicked...")
        self.is_running = True
        self.ms_since_start = 0.0
        self.buffer_a = []
        self.buffer_b = []
        self.active_buffer = self.buffer_a
        self.inactive_buffer = self.buffer_b
        self.chunk_counter = 0
        self.edge_times_deque.clear()
        self.ticks_deque.clear()
        self.noises_deque.clear()
        self.start_btn.disabled = True
        self.stop_btn.disabled = False
        # Disable inputs while running
        self.set_inputs_enabled(False)
        self.status_label.text = "Recording Active..."
        self.stream = None
        self.mic = None

        self.audio_chart.clear_buffers()  # Clear the plot buffers at the start of a new session

        try:
            if platform == 'android':
                # Request audio and storage permissions on Android
                from android.permissions import request_permissions, check_permission, Permission
                perms = [Permission.RECORD_AUDIO, Permission.WRITE_EXTERNAL_STORAGE]
                
                if all(check_permission(p) for p in perms):
                    self.start_android_audio()
                else:
                    self.tell(f"[Android] Requesting permissions {perms}...")
                    request_permissions(perms, self.permission_callback)
                return
            else:
                # Use SoundDevice for Windows/Mac
                import sounddevice as sd
                self.tell("[SoundDevice] Initializing stream...")
                self.stream = sd.InputStream(
                    samplerate=self.sample_rate, 
                    channels=1, 
                    callback=self.audio_callback
                )
                self.stream.start()
                self.tell("[SoundDevice] Stream started")

            self.detector.reset()  # Reset Claude's detector state

            # Schedule the Tier 2 Analysis/Export every 4 seconds
            Clock.schedule_interval(self.process_chunk_buffer, self.chunk_time_s)
            self.tell("[Main] Buffer processing scheduled")
            
        except Exception as e:
            self.tell(f"CRITICAL ERROR: {e}")
            self.status_label.text = f"Mic Error: {e}"
            self.stop_session(None)

    def stop_session(self, instance):
        self.is_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if hasattr(self, 'mic') and self.mic:
            self.mic.stop()
            self.mic = None
        
        Clock.unschedule(self.process_chunk_buffer)
        self.start_btn.disabled = False
        self.stop_btn.disabled = True
        # Re-enable inputs when stopped
        self.set_inputs_enabled(True)
        self.status_label.text = "Stopped. Check project folder for CSVs."

    def process_chunk_buffer(self, dt): # called every chunk_time_s 
        """Swing-buffer handler: Kivy timer is the master clock. Drain, process, and analyze whatever's in the buffer."""
        # SWAP BUFFERS FIRST: Tier 1 immediately switches to the fresh buffer
        self.active_buffer, self.inactive_buffer = self.inactive_buffer, self.active_buffer
        num_samples = len(self.inactive_buffer)
        calc_sample_rate = num_samples/self.chunk_time_s # hopefully close to self.sample_rate (144100 or 8000 Hz)
        sample_rate_error_pc = (calc_sample_rate / self.sample_rate -1)*100 # self.sample_rate is nominal sample clock 
        file_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # wall clock time
        self.tell(f"[pc]     {file_timestamp} : chunk starts at {self.ms_since_start:.3f}", 3)
        self.tell(f"[pc] Samples: {num_samples} in {self.chunk_time_s} s -> {calc_sample_rate} Hz, ({sample_rate_error_pc:.3f} %)", 4)
        
        audio_chunk_amp = np.array(self.inactive_buffer) # copy out pos/neg waveform samples
        self.inactive_buffer.clear()

        # PROCESS THE AUDIO CHUNK ====================================
        time_series_data, chunk_edge_times = self.detector.process_chunk(audio_chunk_amp)
#        time_series_data is {'time_axis', 'audio_chunk', 'onset_strength', 'threshold', 'fast_env', 'slow_env', 'filtered'}

        self.tell(f"[pc] Chunk {self.chunk_counter}, found {len(chunk_edge_times)} edges, total {len(self.edge_times_deque)}", 4)
        self.edge_times_deque.extend(chunk_edge_times) # cross-chunk data accumulation for beat weeding

        # WEEDING: ===================================================
        # Weeding the Window and Updating the Charts with bad beats, ticks and tocks---------------------
        ticks = noises = [] # calculated for whole 'window' - formerly good_nbeats, noises. May later calculate tocks, distinct from ticks
        self.chunk_counter += 1
        if self.chunk_counter % self.chunks_per_window == 0: # time to weed (edge_times_deque is longer than just this chunk)
            ticks, noises, histogram_data = self.detector.weed_edges_in_window(self.edge_times_deque, clock_name='unknown')
            self.tell(f"[pc] Edges are {len(ticks)} Ticks, {len(noises)} Noises", 3)

        # PLOTTING: ==================================================
        # Delayed until after conditionally calculating good and bad beats
        # downsample the time series arrays by self.audio_downsample_factor
        # in the plotter, these are appended to the plotting deques (length self.audio_chart_time_s)

        f = self.audio_downsample_factor
        downsampled_data = {k: v[::f] for k, v in time_series_data.items()} # downsamples ALL members of time_series_data

        selected_ds_data = {'time_axis': downsampled_data['time_axis'], 'onset_strength': downsampled_data['onset_strength'], 'threshold': downsampled_data['threshold']}

        event_data = {'edges': chunk_edge_times, 'ticks': ticks, 'noises': noises}

        self.audio_chart.add_chunk_to_plot(selected_ds_data, event_data) # plot all of them
      
        if False: # 
            # Thread 1: Save raw audio CSV in Tier 2 Worker Thread (Daemon=True so they die if app closes)
            threading.Thread(
                target=write_csv, args=(self, "amps", file_timestamp, 
                        ["Time_ms", "Amplitude", "AbsAmp", "Peaks", "Peaks_dt", "Pulses", "Pulses_dt"], 
                        [audio_chunk_ms, audio_chunk_amp, audio_chunk_abs, 
                        audio_peaks_padded, audio_peaks_padded_dt,
                        audio_pulses_padded, audio_pulses_padded_dt
                        ]
                    ), daemon=True
            ).start()
        
        if False: # 
            # Thread 2: Save just the condensed peaks CSV in Tier 2 Worker Thread (Daemon=True so they die if app closes)
            threading.Thread(
                target=write_csv, args=(self, "peaks", file_timestamp, 
                        ["Time_ms", "Peak", "Peak_dt"], 
                        [audio_peaks_ms, audio_peaks, audio_peaks_dt]
                    ), daemon=True
            ).start()
        
        if False: # 
            # Thread 3: Save just the identified pulses CSV in Tier 2 Worker Thread (Daemon=True so they die if app closes)
            threading.Thread(
                target=write_csv, args=(self, "pulses", file_timestamp, 
                        ["Time_ms", "Pulse", "Pulse_dt"], 
                        [audio_pulses_ms, audio_pulses, audio_pulses_dt]
                    ), daemon=True
            ).start()
        
        if False: # 
            # Thread 4: Compute and save histogram in Tier 2 Worker Thread (Daemon=True so they die if app closes)
            # Prepare histogram data arrays
            bin_indices = np.arange(len(counts))
            bin_maxes = bin_edges[1:]
            percentages = (counts / num_samples * 100)
            
            threading.Thread(
                target=write_csv, args=(self, "bins", file_timestamp, 
                        ["Bin", "Max_Value", "Count", "Percentage"], 
                        [bin_indices, bin_maxes, counts, percentages]
                        ), daemon=True
            ).start()
        
        if False: # 
            # Thread 2: Save the results of Beat Detector to CSV
#            'time_axis': time_axis  # Time in seconds for each sample in chunk
#            'onset_strength': onset_strength,
#            'fast_env': fast_env,
#            'slow_env': slow_env,
#            'filtered': filtered,
#            'threshold': threshold,

            threading.Thread(
                target=write_csv, args=(self, "claude", file_timestamp, 
                        ["Time_ms", "Time_s", "audio", "filtered", "onset_strength"], 
                        [audio_pulses_ms, res['time_axis'], audio_chunk_amp, res['filtered'], res['onset_strength']]
                    ), daemon=True
            ).start()
        
        self.status_label.text = f"Recording... Last CSV: {file_timestamp}"

        # Update the running timestamp counter
        self.ms_since_start += (len(audio_chunk_amp) / self.sample_rate) * 1000.0


if __name__ == '__main__':
    ClockApp().run()