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

# --- TIER 2: ANALYSIS & I/O WORKER ---
def save_csv_worker(app, data_chunk, timestamp):
    """Writes the 4s buffer to a CSV file in a background thread."""
    filename = f"clock_log_{timestamp}.csv"
    try:
        # Use Android-safe directory path
        filepath = os.path.join(App.get_running_app().user_data_dir, filename)
        # We use a context manager to ensure the file closes properly
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Amplitude"])
            # Writing as a column for easy Excel analysis
            writer.writerows([[val] for val in data_chunk])
        app.tell(f"[Tier 2] Successfully saved {len(data_chunk)} samples to {filename}")
    except Exception as e:
        app.tell(f"[Tier 2] I/O Error: {e}")

def save_histogram_worker(app, data_chunk, timestamp):
    """Computes and writes histogram binning to a CSV file in a background thread."""
    filename = f"bins_{timestamp}.csv"
    try:
        # Use Android-safe directory path
        filepath = os.path.join(App.get_running_app().user_data_dir, filename)
        # Convert to numpy array and take absolute values
        data_array = np.abs(np.array(data_chunk))
        max_val = np.max(data_array) if len(data_array) > 0 else 1.0
        total_samples = len(data_array)
        
        # Create 100 bins from 0 to max_val
        counts, bin_edges = np.histogram(data_array, bins=100, range=(0, max_val))
        
        # Write histogram to CSV: bin_number, max_value_of_bin, count, percentage
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Bin", "Max_Value", "Count", "Percentage"])
            for bin_num, (bin_max, count) in enumerate(zip(bin_edges[1:], counts)):
                percentage = (count / total_samples * 100) if total_samples > 0 else 0.0
                writer.writerow([bin_num, bin_max, int(count), f"{percentage:.2f}"])
        app.tell(f"[Tier 2] Successfully saved histogram to {filename}")
    except Exception as e:
        app.tell(f"[Tier 2] Histogram Error: {e}")

class ClockApp(App):
    def build(self):
        # Configuration
        self.sample_rate = 44100
        self.window_duration = 4.0 
        self.samples_per_window = int(self.sample_rate * self.window_duration)
        
        # State Management
        self.is_running = False
        self.stream = None
        # Double-buffer (swing buffer) for lock-free operation
        self.buffer_a = []
        self.buffer_b = []
        self.active_buffer = self.buffer_a  # Tier 1 appends to this
        self.inactive_buffer = self.buffer_b  # Tier 2 processes this
        
        # UI Setup
        layout = BoxLayout(orientation='vertical', padding=30, spacing=20)
        
        # Buttons layout - horizontal at top
        buttons_layout = BoxLayout(orientation='horizontal', size_hint_y=0.15, spacing=10)
        
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

        # Status label
        self.status_label = Label(
            text="Ready to Record. Press Start.", 
            halign='center', font_size='12sp',
            size_hint_y=0.15
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
    def tell(self, message):
        """Write a message to the scrolling text region and optionally to console."""
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

    def start_session(self, instance):
        self.tell("Start button clicked...")
        self.is_running = True
        self.buffer_a = []
        self.buffer_b = []
        self.active_buffer = self.buffer_a
        self.inactive_buffer = self.buffer_b
        self.start_btn.disabled = True
        self.stop_btn.disabled = False
        self.status_label.text = "Recording Active..."
        self.stream = None
        self.mic = None

        try:
            if platform == 'android':
                # Request audio permissions on Android
                from android.permissions import request_permissions, Permission
                request_permissions([Permission.RECORD_AUDIO])
                self.tell("[Android] Audio permission requested")
                
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
            timestamp = datetime.now().strftime("%H-%M-%S")
            
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