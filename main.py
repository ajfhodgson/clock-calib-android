import numpy as np
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

# Mask bits for tell() messages:
# Bit 0 (1): Startup and status/error messages
# Bit 1 (2): 
# Bit 2 (4): File writing messages
# Bit 3 (8): Chunk Analysis summary
# Bit 4 (16): Chunk Analysis detail 
# 

#&&ToDo - avoid the crash after first run/request for microphone permission on Android
#&&ToDo - ensure there is a timestamp (ms since pressing Start) with the amplitude data in CSV
#&&ToDO - Generate an array of peaks with their timestamps for later analysis
#&&ToDo - Add a scrolling plot of the received amplitudes, marking the detected peaks
#&&ToDo - numpy.Autocorellate to detect frequency of the pulse trains (should also indicate beat error)
#&&ToDo - Decide whether each peak is a 'tick' or a 'tock' based on timing


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
    filename = f"{filename_root}_{timestamp}.csv"
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
    def build(self):
        if platform != 'android':
            Window.size = (352, 750)

        # Configuration
        # defaults
        self.target_bph = 8860 # sriaght-sided mantel clock
        self.peak_thresh_pc = 0.1
        self.sample_rate = 44100
        self.window_duration = 4.0 
        self.samples_per_window = int(self.sample_rate * self.window_duration)
        self.tell_mask = 31         # Controls which tell() messages are emitted; bitmask flags (default: just bit 0 = 1)
        
        # State Management
        self.is_running = False
        self.stream = None
        self.ms_since_start = 0.0  # running ms counter (dead-reckoning since Start pressed)

        # Double-buffer (swing buffer) for lock-free operation
        self.buffer_a = []
        self.buffer_b = []
        self.active_buffer = self.buffer_a  # Tier 1 appends to this
        self.inactive_buffer = self.buffer_b  # Tier 2 processes this
        
        # UI Setup
        layout = BoxLayout(orientation='vertical', padding=30, spacing=20)
        
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

        # Window Size (seconds)
        ws_box = BoxLayout(orientation='horizontal')
        l1 = Label(text='Chunk\n(sec)', size_hint_x=1.0)
        autoscale(l1, 0.3)
        ws_box.add_widget(l1)

        self.win_input = TextInput(text=str(self.window_duration), multiline=False, input_filter='float', size_hint_x=0.9)
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

        # Scrolling text region (40% of screen)
        self.log_scroll = ScrollView(
            size_hint_y=0.4,
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

        # Spacer to push everything to top
        spacer = Label(
            size_hint_y=1,
            text='reserved for future use'
        )
        layout.add_widget(spacer)

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

    # --- TIER 1: HIGH-PRIORITY LISTENER ---
    def audio_callback(self, indata, frames, time_info, status):
        """Hardware-level callback. Minimal logic to avoid underruns."""
        if self.is_running:
            # Append to whichever buffer is currently active.
            # No lock needed: Tier 1 only appends to active_buffer,
            # and Tier 2 only reads from inactive_buffer.
            samples_count = len(indata.flatten())
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

        if platform != 'android':
            print(message)
        
        # Schedule the UI update on the main thread
        Clock.schedule_once(lambda dt: self._update_log( str(mask_bit) + ': ' + message ), 0)
    
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

    def _set_window_duration(self, text):
        val = self._parse_float(text, self.window_duration)
        if val <= 0:
            self.tell("[UI] Invalid window duration, keeping previous value")
            self.win_input.text = str(self.window_duration)
            return
        self.window_duration = val
        self.samples_per_window = int(self.sample_rate * self.window_duration)
        self.tell(f"[UI] Window duration set to {self.window_duration}s")
        if self.is_running:
            Clock.unschedule(self.process_buffer)
            Clock.schedule_interval(self.process_buffer, self.window_duration)
            self.tell("[UI] Rescheduled buffer processing to new window duration")

    def on_window_input(self, instance):
        self._set_window_duration(instance.text)

    def on_window_focus(self, instance, value):
        if not value:
            self._set_window_duration(instance.text)

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
            
            Clock.schedule_interval(self.process_buffer, self.window_duration)
            self.tell("[Main] Buffer processing scheduled")
        except Exception as e:
            self.tell(f"CRITICAL ERROR: {e}")
            self.status_label.text = f"Mic Error: {e}"
            self.stop_session(None)

    def permission_callback(self, permissions, results):
        if all(results):
            self.tell("[Android] Permissions granted.")
            self.start_android_audio()
        else:
            self.tell("[Android] Permissions denied.")
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
        self.start_btn.disabled = True
        self.stop_btn.disabled = False
        # Disable inputs while running
        self.set_inputs_enabled(False)
        self.status_label.text = "Recording Active..."
        self.stream = None
        self.mic = None

        try:
            if platform == 'android':
                # Request audio and storage permissions on Android
                from android.permissions import request_permissions, check_permission, Permission
                perms = [Permission.RECORD_AUDIO, Permission.WRITE_EXTERNAL_STORAGE]
                
                if all(check_permission(p) for p in perms):
                    self.start_android_audio()
                else:
                    self.tell("[Android] Requesting permissions...")
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

            # Schedule the Tier 2 Analysis/Export every 4 seconds
            Clock.schedule_interval(self.process_buffer, self.window_duration)
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
        
        Clock.unschedule(self.process_buffer)
        self.start_btn.disabled = False
        self.stop_btn.disabled = True
        # Re-enable inputs when stopped
        self.set_inputs_enabled(True)
        self.status_label.text = "Stopped. Check project folder for CSVs."

    def process_buffer(self, dt):
        """Swing-buffer handler: Kivy timer is the master clock. Drain, process, and analyze whatever's in the buffer."""
        # SWAP BUFFERS FIRST: Tier 1 immediately switches to the fresh buffer
        self.active_buffer, self.inactive_buffer = self.inactive_buffer, self.active_buffer
        max_val = max(self.inactive_buffer) if self.inactive_buffer else 0
        num_samples = len(self.inactive_buffer)
        self.tell(f"[Tier 2] Chunk: {num_samples} samples at {len(self.inactive_buffer)/self.window_duration} Hz, (max: {max_val:.4f})", 3)
        
        # Take all accumulated data from the (now-frozen) inactive buffer
        audio_chunk_amp = np.array(self.inactive_buffer)
        self.inactive_buffer.clear()
        file_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # wall clock time
        if num_samples > 0:
            # calculate the ms timestamps for each samle (since Start was pressed)
            audio_chunk_ms = self.ms_since_start + np.arange(len(audio_chunk_amp)) * (1000.0 / self.sample_rate)

            # find peaks - a) what amplitude is exceeded by only peak_thresh_pc % of samples?
            audio_chunk_abs = np.abs(audio_chunk_amp)
            # Create 100 bins from 0 to max_val
            counts, bin_edges = np.histogram(audio_chunk_abs, bins=100, range=(0, max_val))
            
            cumulative_percentage = 0.0 
            for bin_num, (bin_max, count) in enumerate(zip(bin_edges[1:], counts)):
                percentage = (count / num_samples * 100)
                cumulative_percentage += percentage
                if cumulative_percentage >= (100.0 - self.peak_thresh_pc):
                    peak_threshold = bin_max
                    break # found our threshold
            self.tell(f"[Tier 2] Chunk: {self.peak_thresh_pc}% of samples exceeded {peak_threshold:.4f}", 4)

            mask = audio_chunk_abs >= peak_threshold  # identify peaks
            audio_peaks = audio_chunk_abs[mask] # condensed array of just the peaks
            audio_peaks_ms = audio_chunk_ms[mask] # and the associated times
            audio_peaks_dt = np.diff(audio_peaks_ms, prepend=audio_peaks_ms[0]) 

            audio_peaks_padded = np.where(mask, audio_chunk_abs, 0.0) # full lenth array of just peaks
            audio_peaks_padded_dt = np.zeros_like(audio_chunk_ms) # and the associated dt since last peak
            if len(audio_peaks_ms) > 0:
                audio_peaks_padded_dt[mask] = audio_peaks_dt
            
           
            audio_average = np.mean(audio_chunk_abs) if len(audio_chunk_abs) > 0 else 0.0
            peaks_average = np.mean(audio_peaks) if len(audio_peaks) > 0 else 0.0

            self.tell(f"[Tier 2] Peaks: {len(audio_peaks)}, Max: {max_val:.4f}, Avg Peak: {peaks_average:.4f}, Audio Avg: {audio_average:.4f}", 4)
            
            # Thread 1: Save raw audio CSV in Tier 2 Worker Thread (Daemon=True so they die if app closes)
            # columns: [Time_ms, Amplitude]
            threading.Thread(
                target=write_csv, 
                args=(self, "amps", file_timestamp, 
                      ["Time_ms", "Amplitude", "AbsAmp", "Peaks", "Peaks_DT"], 
                      [audio_chunk_ms, audio_chunk_amp, audio_chunk_abs, 
                       audio_peaks_padded, audio_peaks_padded_dt]
                    ), 
                daemon=True
            ).start()
            
            # Thread 2: Save just the condensed peaks CSV in Tier 2 Worker Thread (Daemon=True so they die if app closes)
            # columns: [Time_ms, Amplitude]
            threading.Thread(
                target=write_csv, 
                args=(self, "peaks", file_timestamp, 
                      ["Time_ms", "Peak", "Peak_DT"], 
                      [audio_peaks_ms, audio_peaks, audio_peaks_dt]
                    ), 
                daemon=True
            ).start()
            
            # Thread 3: Compute and save histogram in Tier 2 Worker Thread (Daemon=True so they die if app closes)
            # Prepare histogram data arrays
            bin_indices = np.arange(len(counts))
            bin_maxes = bin_edges[1:]
            percentages = (counts / num_samples * 100)
            
            threading.Thread(
                target=write_csv, 
                args=(self, "bins", file_timestamp, ["Bin", "Max_Value", "Count", "Percentage"], [bin_indices, bin_maxes, counts, percentages]), 
                daemon=True
            ).start()
            
            self.status_label.text = f"Recording... Last CSV: {file_timestamp}"

            # Update the running timestamp counter
            self.ms_since_start += (len(audio_chunk_amp) / self.sample_rate) * 1000.0

if __name__ == '__main__':
    ClockApp().run()