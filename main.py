import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.clock import Clock
from kivy.utils import platform
import threading
import csv
import os
from datetime import datetime

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

def verify_and_report_file(app, filepath, num_samples=None):
    """Verify file exists, get size, and report via app.tell()."""
    if os.path.exists(filepath):
        file_size = os.path.getsize(filepath)
        if file_size > 0:
            if num_samples is not None:
                app.tell(f"[Tier 2] Successfully saved {num_samples} samples to {filepath} ({file_size} bytes)")
            else:
                app.tell(f"[Tier 2] Successfully saved to {filepath} ({file_size} bytes)")
        else:
            app.tell(f"[Tier 2] ERROR: File created but empty - {filepath} (0 bytes)")
    else:
        app.tell(f"[Tier 2] ERROR: File does not exist - {filepath}")

# --- TIER 2: ANALYSIS & I/O WORKER ---
def save_csv_worker(app, data_chunk, timestamp):
    """Writes the 4s buffer to a CSV file in a background thread."""
    filename = f"amps_{timestamp}.csv"
    try:
        base_dir = get_save_directory()
        filepath = os.path.join(base_dir, filename)

        # Write CSV with time and amplitude columns
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header with time (ms since program start) then amplitude
            writer.writerow(["Time_ms", "Amplitude"])
            # Dead-reckon times from app.samples_written using app.sample_rate
            rows = []
            sr = getattr(app, 'sample_rate', 44100)
            start_idx = getattr(app, 'samples_written', 0)
            for i, val in enumerate(data_chunk):
                # time in milliseconds since app start, formatted to 3 decimal places
                time_ms = (start_idx + i) / float(sr) * 1000.0
                rows.append([f"{time_ms:.3f}", val])
            writer.writerows(rows)
            # Update global counter (approximate / dead-reckoned)
            try:
                app.samples_written = start_idx + len(data_chunk)
            except Exception:
                pass

        # Verify and report file
        verify_and_report_file(app, filepath, num_samples=len(data_chunk))
    except Exception as e:
        app.tell(f"[Tier 2] I/O Error: {e}")

def save_histogram_worker(app, data_chunk, timestamp):
    """Computes and writes histogram binning to a CSV file in a background thread."""
    filename = f"bins_{timestamp}.csv"
    try:
        base_dir = get_save_directory()
        filepath = os.path.join(base_dir, filename)

        # Convert to numpy array and take absolute values
        data_array = np.abs(np.array(data_chunk))
        max_val = np.max(data_array) if len(data_array) > 0 else 1.0
        total_samples = len(data_array)
        
        # Create 100 bins from 0 to max_val
        counts, bin_edges = np.histogram(data_array, bins=100, range=(0, max_val))
        
        # Write histogram to CSV
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Bin", "Max_Value", "Count", "Percentage"])
            for bin_num, (bin_max, count) in enumerate(zip(bin_edges[1:], counts)):
                percentage = (count / total_samples * 100) if total_samples > 0 else 0.0
                writer.writerow([bin_num, bin_max, int(count), f"{percentage:.2f}"])

        # Verify and report file
        verify_and_report_file(app, filepath)
    except Exception as e:
        app.tell(f"[Tier 2] Histogram Error: {e}")

#############################################-----------------------------------------------

class ClockApp(App):
    def build(self):
        # Configuration
        # defaults
        self.target_bph = 0
        self.peak_thresh_pc = 0.1
        self.sample_rate = 44100
        self.window_duration = 4.0 
        self.samples_per_window = int(self.sample_rate * self.window_duration)

        # Controls which tell() messages are emitted; bitmask flags (default: just bit 0 = 1)
        self.tell_mask = 1
        
        # State Management
        self.is_running = False
        self.stream = None
        # running sample counter (dead-reckoning since app start)
        self.samples_written = 0
        # Double-buffer (swing buffer) for lock-free operation
        self.buffer_a = []
        self.buffer_b = []
        self.active_buffer = self.buffer_a  # Tier 1 appends to this
        self.inactive_buffer = self.buffer_b  # Tier 2 processes this
        
        # UI Setup
        layout = BoxLayout(orientation='vertical', padding=30, spacing=20)
        
        # Buttons layout - horizontal at top (10% height)
        buttons_layout = BoxLayout(orientation='horizontal', size_hint_y=0.10, spacing=10)
        
        self.start_btn = Button(text="Start", size_hint_x=1)
        self.start_btn.bind(on_press=self.start_session)
        buttons_layout.add_widget(self.start_btn)

        self.stop_btn = Button(text="Stop", disabled=True, size_hint_x=1)
        self.stop_btn.bind(on_press=self.stop_session)
        buttons_layout.add_widget(self.stop_btn)

        self.exit_btn = Button(text="Exit", size_hint_x=1)
        self.exit_btn.bind(on_press=self.exit_app)
        buttons_layout.add_widget(self.exit_btn)
        
        layout.add_widget(buttons_layout)

        # Entry fields row - horizontal (10% height)
        entries_layout = BoxLayout(orientation='horizontal', size_hint_y=0.10, spacing=10)

        # Window Size (seconds)
        ws_box = BoxLayout(orientation='vertical')
        ws_box.add_widget(Label(text='Window Size (s)', size_hint_y=None, height='20dp'))
        self.win_input = TextInput(text=str(self.window_duration), multiline=False, input_filter='float', size_hint_y=None, height='36dp')
        self.win_input.bind(on_text_validate=self.on_window_input, on_focus=self.on_window_focus)
        ws_box.add_widget(self.win_input)
        entries_layout.add_widget(ws_box)

        # Target BPH
        tb_box = BoxLayout(orientation='vertical')
        tb_box.add_widget(Label(text='Target BPH', size_hint_y=None, height='20dp'))
        self.bph_input = TextInput(text=str(getattr(self, 'target_bph', 0)), multiline=False, input_filter='int', size_hint_y=None, height='36dp')
        self.bph_input.bind(on_text_validate=self.on_bph_input, on_focus=self.on_bph_focus)
        tb_box.add_widget(self.bph_input)
        entries_layout.add_widget(tb_box)

        # Peak Threshold %
        pt_box = BoxLayout(orientation='vertical')
        pt_box.add_widget(Label(text='Peak Threshold %', size_hint_y=None, height='20dp'))
        self.peak_input = TextInput(text=str(getattr(self, 'peak_thresh_pc', 0.1)), multiline=False, input_filter='float', size_hint_y=None, height='36dp')
        self.peak_input.bind(on_text_validate=self.on_peak_input, on_focus=self.on_peak_focus)
        pt_box.add_widget(self.peak_input)
        entries_layout.add_widget(pt_box)

        # Tell mask (bitmask input)
        ts_box = BoxLayout(orientation='vertical')
        ts_box.add_widget(Label(text='Tell mask', size_hint_y=None, height='20dp'))
        self.tell_input = TextInput(text=str(getattr(self, 'tell_mask', 1)), multiline=False, input_filter='int', size_hint_y=None, height='36dp')
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
        self.log_text = TextInput(
            text='',
            readonly=True,
            font_size='12sp',
            size_hint_y=0.4,
            foreground_color=(1, 1, 1, 1),  # White text
            background_color=(0.2, 0.2, 0.2, 1)  # Dark grey background
        )
        layout.add_widget(self.log_text)

        # Spacer to push everything to top
        spacer = Label(
            size_hint_y=1,
            text='reserved for future use'
        )
        layout.add_widget(spacer)

        return layout

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
        # If the requested flag bit(s) are not set in tell_mask, skip output
        try:
            if not ((1 << mask_bit) & getattr(self, 'tell_mask', 1)):
                return
        except Exception:
            pass

        if platform != 'android':
            print(message)
        
        # Schedule the UI update on the main thread
        Clock.schedule_once(lambda dt: self._update_log(message), 0)
    
    def _update_log(self, message):
        """Internal method to update the log text (runs on main thread)."""
        # Add to text region with newline
        if self.log_text.text:
            self.log_text.text += '\n' + message
        else:
            self.log_text.text = message
        
        # Scroll to bottom (scroll_y=0 is bottom in Kivy)
        self.log_text.scroll_y = 0

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
        self.tell(f"[UI] Tell mask set to {self.tell_mask}", flag=1)

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

    def start_session(self, instance):
        self.tell("Start button clicked...")
        self.is_running = True
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
                from android.permissions import request_permissions, Permission
                request_permissions([Permission.RECORD_AUDIO, Permission.WRITE_EXTERNAL_STORAGE])
                self.tell("[Android] Audio and storage permissions requested")
                
                # Use the custom Android class we built
                from android_audio import AndroidMic
                self.mic = AndroidMic(sample_rate=self.sample_rate)
                self.tell("[Android] AndroidMic initialized, starting callback loop...")
                # Pass the callback to the Android version
                self.mic.start(callback=self.audio_callback)
                self.tell("[Android] AndroidMic started")
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
        
        # Calculate maximum value in the buffer
        max_val = max(self.inactive_buffer) if self.inactive_buffer else 0
        
        self.tell(f"[Tier 2] Buffer swap: processing {len(self.inactive_buffer)} samples from inactive buffer (max: {max_val})")
        
        # Take all accumulated data from the (now-frozen) inactive buffer
        chunk_to_save = self.inactive_buffer.copy()
        
        # Clear the inactive buffer for the next cycle
        self.inactive_buffer.clear()
        
        # Only export if there's data to write
        if chunk_to_save:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            # Spin up Tier 2 Worker Threads (Daemon=True so they die if app closes)
            # Thread 1: Save raw audio CSV
            threading.Thread(
                target=save_csv_worker, 
                args=(self, chunk_to_save, timestamp), 
                daemon=True
            ).start()
            
            # Thread 2: Compute and save histogram
            threading.Thread(
                target=save_histogram_worker, 
                args=(self, chunk_to_save, timestamp), 
                daemon=True
            ).start()
            
            self.status_label.text = f"Recording... Last CSV: {timestamp}"

if __name__ == '__main__':
    ClockApp().run()