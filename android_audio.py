from jnius import autoclass
from android.permissions import request_permissions, Permission

# Load the necessary Java classes from the Android API
AudioRecord = autoclass('android.media.AudioRecord')
AudioFormat = autoclass('android.media.AudioFormat')
MediaRecorder = autoclass('android.media.MediaRecorder')
AudioSource = autoclass('android.media.MediaRecorder$AudioSource') # Note the $ sign

class AndroidMic:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        # Define the buffer size for the hardware
        self.buffer_size = AudioRecord.getMinBufferSize(
            sample_rate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )

        # Initialize the recorder
        self.recorder = AudioRecord(
            AudioSource.MIC,  # Use the new AudioSource variable directly
            sample_rate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            self.buffer_size
        )
        

    def start(self):
        self.recorder.startRecording()

    def read(self):
        # Create a Python list (buffer) to hold the audio data
        # Note: Android uses 16-bit integers (shorts)
        buffer = [0] * (self.buffer_size // 2)
        self.recorder.read(buffer, 0, len(buffer))
        return buffer

    def stop(self):
        self.recorder.stop()
        self.recorder.release()
