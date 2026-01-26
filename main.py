import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.utils import platform 
if platform == 'android':
    from android_audio import AndroidMic  # homebrew shim for android to look like Windows sounddevice
    from android.permissions import request_permissions, Permission # type: ignore
    request_permissions([Permission.RECORD_AUDIO])
    mic = AndroidMic(sample_rate=44100)
    mic.start()
else:
    import sounddevice as sd
import threading
import csv
from datetime import datetime

# --- TIER 2: ANALYSIS & I/O WORKER ---
def save_csv_worker(data_chunk, timestamp):
    """Writes the 4s buffer to a CSV file in a background thread."""
    filename = f"clock_log_{timestamp}.csv"
    try:
        # We use a context manager to ensure the file closes properly
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Amplitude"])
            # Writing as a column for easy Excel analysis
            writer.writerows([[val] for val in data_chunk])
        print(f"[Tier 2] Successfully saved {len(data_chunk)} samples to {filename}")
    except Exception as e:
        print(f"[Tier 2] I/O Error: {e}")

def save_histogram_worker(data_chunk, timestamp):
    """Computes and writes histogram binning to a CSV file in a background thread."""
    filename = f"bins_{timestamp}.csv"
    try:
        # Convert to numpy array and take absolute values
        data_array = np.abs(np.array(data_chunk))
        max_val = np.max(data_array) if len(data_array) > 0 else 1.0
        total_samples = len(data_array)
        
        # Create 100 bins from 0 to max_val
        counts, bin_edges = np.histogram(data_array, bins=100, range=(0, max_val))
        
        # Write histogram to CSV: bin_number, max_value_of_bin, count, percentage
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Bin", "Max_Value", "Count", "Percentage"])
            for bin_num, (bin_max, count) in enumerate(zip(bin_edges[1:], counts)):
                percentage = (count / total_samples * 100) if total_samples > 0 else 0.0
                writer.writerow([bin_num, bin_max, int(count), f"{percentage:.2f}"])
        print(f"[Tier 2] Successfully saved histogram to {filename}")
    except Exception as e:
        print(f"[Tier 2] Histogram Error: {e}")

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
        self.status_label = Label(
            text="Ready to Record\nCSV logs will save every 4s", 
            halign='center', font_size='18sp'
        )
        layout.add_widget(self.status_label)

        self.start_btn = Button(text="START RECORDING", size_hint_y=0.4)
        self.start_btn.bind(on_press=self.start_session)
        layout.add_widget(self.start_btn)

        self.stop_btn = Button(text="STOP", size_hint_y=0.4, disabled=True)
        self.stop_btn.bind(on_press=self.stop_session)
        layout.add_widget(self.stop_btn)

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
    def start_session(self, instance):
        self.is_running = True
        self.buffer_a = []
        self.buffer_b = []
        self.active_buffer = self.buffer_a
        self.inactive_buffer = self.buffer_b
        self.start_btn.disabled = True
        self.stop_btn.disabled = False
        self.status_label.text = "Recording Active..."

        # Initialize and start SoundDevice stream
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate, 
                channels=1, 
                callback=self.audio_callback
            )
            self.stream.start()
            # Schedule the Tier 2 Analysis/Export every 4 seconds
            Clock.schedule_interval(self.process_buffer, self.window_duration)
        except Exception as e:
            self.status_label.text = f"Mic Error: {e}"
            self.stop_session(None)

    def stop_session(self, instance):
        self.is_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        Clock.unschedule(self.process_buffer)
        self.start_btn.disabled = False
        self.stop_btn.disabled = True
        self.status_label.text = "Stopped. Check project folder for CSVs."

    def process_buffer(self, dt):
        """Swing-buffer handler: Kivy timer is the master clock. Drain, process, and analyze whatever's in the buffer."""
        # SWAP BUFFERS FIRST: Tier 1 immediately switches to the fresh buffer
        self.active_buffer, self.inactive_buffer = self.inactive_buffer, self.active_buffer
        print(f"[Tier 2] Buffer swap: active buffer now has {len(self.active_buffer)} samples, processing {len(self.inactive_buffer)} from inactive")
        
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
                args=(chunk_to_save, timestamp), 
                daemon=True
            ).start()
            
            # Thread 2: Compute and save histogram
            threading.Thread(
                target=save_histogram_worker, 
                args=(chunk_to_save, timestamp), 
                daemon=True
            ).start()
            
            self.status_label.text = f"Recording...\nLast CSV: {timestamp}"

if __name__ == '__main__':
    ClockApp().run()