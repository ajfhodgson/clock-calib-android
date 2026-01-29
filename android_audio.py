from jnius import autoclass
from android.permissions import request_permissions, Permission
import threading
import numpy as np
import time

# Load the necessary Java classes from the Android API
AudioRecord = autoclass('android.media.AudioRecord')
AudioFormat = autoclass('android.media.AudioFormat')
MediaRecorder = autoclass('android.media.MediaRecorder')
AudioSource = autoclass('android.media.MediaRecorder$AudioSource') # Note the $ sign

class AndroidMic:
    def __init__(self, sample_rate=44100):
        print("PYTHON: Initializing AudioSource...")
        self.sample_rate = sample_rate
        self.is_recording = False
        self.callback = None
        self.read_thread = None

        # Define the buffer size for the hardware
        print("PYTHON: Getting Min Buffer Size...")
        self.buffer_size = AudioRecord.getMinBufferSize(
            sample_rate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )

        # Initialize the recorder
        print(f"PYTHON: Creating AudioRecord object (Buffer: {self.buffer_size})...")
        self.recorder = AudioRecord(
            AudioSource.MIC,  # Use the new AudioSource variable directly
            sample_rate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            self.buffer_size
        )
        print("PYTHON: AudioRecord created successfully!")
        

    def start(self, callback=None):
        """Start recording and feeding data to the callback function."""
        self.callback = callback
        self.is_recording = True
        self.recorder.startRecording()
        
        # Start a background thread to read from AudioRecord and call the callback
        self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.read_thread.start()
        print("PYTHON: Read thread started")

    def _read_loop(self):
        """Background thread that continuously reads from AudioRecord and calls the callback."""
        while self.is_recording:
            try:
                # Read audio data from the recorder
                buffer = [0] * (self.buffer_size // 2)
                self.recorder.read(buffer, 0, len(buffer))
                
                # Convert to numpy array (as int16, matching PCM_16BIT)
                audio_data = np.array(buffer, dtype=np.int16)
                
                # Normalize to float32 in range -1.0..1.0 (matching sounddevice format)
                audio_data = audio_data.astype(np.float32) / 32768.0
                
                # Reshape to match sounddevice format: (frames, channels)
                audio_data = audio_data.reshape(-1, 1)
                
                # Call the callback if provided
                if self.callback:
                    # Create a mock time_info object
                    class TimeInfo:
                        def __init__(self):
                            self.input_buffer_adc_time = time.time()
                    
                    # Call the callback with the data (matching sounddevice signature)
                    self.callback(audio_data, len(buffer), TimeInfo(), None)
                    
            except Exception as e:
                print(f"PYTHON: Error in read loop: {e}")
                if self.is_recording:
                    time.sleep(0.1)  # Brief pause before retrying

    def stop(self):
        """Stop recording and clean up resources."""
        self.is_recording = False
        if self.read_thread:
            self.read_thread.join(timeout=1.0)
        self.recorder.stop()
        self.recorder.release()
        print("PYTHON: Recorder stopped and released")
