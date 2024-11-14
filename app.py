from flask import Flask, render_template, jsonify, send_file, request
import pyaudio
import wave
import threading
import os
import time
import numpy as np
from scipy import signal
from scipy.fft import fft

app = Flask(__name__)


class AudioFilter:
    def __init__(self):
        self.sample_rate = 44100
        self.filter_enabled = False
        self.threshold = 0.02
        self.filter_strength = 0.5
        self.chunk_size = 512

    def set_filter_strength(self, strength):
        self.filter_strength = max(0.0, min(1.0, strength))

    def set_threshold(self, threshold):
        self.threshold = max(0.0, min(1.0, threshold))

    def apply_threshold(self, audio_data):
        normalized = audio_data / 32768.0
        mask = np.abs(normalized) < self.threshold
        normalized[mask] = 0
        return (normalized * 32768).astype(np.int16)

    def apply_filter(self, data):
        try:
            audio_data = np.frombuffer(data, dtype=np.int16)
            audio_data = self.apply_threshold(audio_data)

            if self.filter_enabled:
                nyquist = self.sample_rate / 2
                low_cut = 20 / nyquist
                b_low, a_low = signal.butter(2, low_cut, btype='high')
                high_cut = 22000 / nyquist
                b_high, a_high = signal.butter(2, high_cut, btype='low')

                filtered_data = audio_data.astype(np.float64)
                filtered_low = signal.filtfilt(b_low, a_low, filtered_data)
                filtered_high = signal.filtfilt(b_high, a_high, filtered_low)

                mixed_data = (filtered_high * self.filter_strength +
                              audio_data * (1 - self.filter_strength))

                window_size = 5
                smoothing_window = np.hanning(window_size)
                smoothing_window = smoothing_window / smoothing_window.sum()
                mixed_data = np.convolve(mixed_data, smoothing_window, mode='same')

                return mixed_data.astype(np.int16).tobytes()
            return audio_data.tobytes()
        except Exception as e:
            print(f"Filter error: {str(e)}")
            return data


class AudioRecorder:
    def __init__(self):
        self.is_recording = False
        self.is_playing = False
        self.frames = []
        self.filename = "recorded_audio.wav"
        self.filtered_filename = "filtered_audio.wav"
        self.start_time = None
        self.duration = 0
        self.sample_rate = 44100
        self.chunk_size = 512
        self.update_interval = 50
        self.audio_filter = AudioFilter()
        self.signal_data = []
        self.frequency_data = []
        self.last_update_time = 0
        self.max_frequency = 0  # 添加最大频率跟踪
        self.min_frequency = float('inf')  # 添加最小频率跟踪

    def get_frequency_spectrum(self, data):
        try:
            audio_data = np.frombuffer(data, dtype=np.int16)
            window = np.hanning(len(audio_data))
            windowed_data = audio_data * window

            n_fft = min(2048, len(audio_data))
            fft_data = fft(windowed_data[:n_fft])

            freq = np.fft.fftfreq(n_fft, 1 / self.sample_rate)
            positive_mask = freq > 0
            freq = freq[positive_mask]
            magnitude = np.abs(fft_data[positive_mask])

            magnitude_threshold = np.max(magnitude) * 0.1
            peaks = magnitude > magnitude_threshold

            if np.any(peaks):
                max_idx = np.argmax(magnitude)
                dominant_freq = freq[max_idx]
                dominant_freq = min(max(20, dominant_freq), 20000)

                # 更新频率范围
                if dominant_freq > self.max_frequency:
                    self.max_frequency = dominant_freq
                if dominant_freq < self.min_frequency:
                    self.min_frequency = dominant_freq

                return dominant_freq
            return 0

        except Exception as e:
            print(f"FFT error: {str(e)}")
            return 0

    def _record_audio(self):
        try:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )

            while self.is_recording:
                try:
                    data = stream.read(self.chunk_size)
                    current_time = time.time() * 1000

                    if current_time - self.last_update_time >= self.update_interval:
                        audio_data = np.frombuffer(data, dtype=np.int16)
                        normalized = np.abs(audio_data / (32768.0 * 2))
                        strength = -100
                        if np.mean(normalized) > 0:
                            strength = 20 * np.log10(np.mean(normalized))
                        self.signal_data.append(strength)

                        freq = self.get_frequency_spectrum(data)
                        if freq > 0:
                            self.frequency_data.append(freq)

                        self.last_update_time = current_time

                    if self.audio_filter.filter_enabled:
                        data = self.audio_filter.apply_filter(data)
                    self.frames.append(data)

                except Exception as e:
                    print(f"Recording error: {str(e)}")
                    break

            stream.stop_stream()
            stream.close()
            p.terminate()

            self.duration = int(time.time() - self.start_time)

            if len(self.frames) > 0:
                output_filename = self.filtered_filename if self.audio_filter.filter_enabled else self.filename
                wf = wave.open(output_filename, 'wb')
                wf.setnchannels(1)
                wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.frames))
                wf.close()

        except Exception as e:
            print(f"Recording error: {str(e)}")
            self.is_recording = False

    def start_recording(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)
        if os.path.exists(self.filtered_filename):
            os.remove(self.filtered_filename)

        self.is_recording = True
        self.frames = []
        self.signal_data = []
        self.frequency_data = []
        self.max_frequency = 0  # 重置频率范围
        self.min_frequency = float('inf')
        self.start_time = time.time()
        threading.Thread(target=self._record_audio, daemon=True).start()

    def get_frequency_range(self):
        """获取当前频率范围"""
        if self.min_frequency == float('inf'):
            return 0, 22000  # 默认范围

        min_freq = max(20, self.min_frequency * 0.8)  # 留出一些余量
        max_freq = min(22000, self.max_frequency * 1.2)

        return min_freq, max_freq


recorder = AudioRecorder()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/start_recording', methods=['POST'])
def start_recording():
    try:
        if not recorder.is_recording:
            recorder.start_recording()
            return jsonify({"status": "success"})
        return jsonify({"status": "already recording"})
    except Exception as e:
        print(f"Start recording error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    try:
        if recorder.is_recording:
            recorder.is_recording = False
            return jsonify({"status": "success"})
        return jsonify({"status": "not recording"})
    except Exception as e:
        print(f"Stop recording error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/get_recording')
def get_recording():
    filename = recorder.filtered_filename if recorder.audio_filter.filter_enabled else recorder.filename
    if os.path.exists(filename):
        return send_file(filename)
    return jsonify({"status": "no recording found"})


@app.route('/get_audio_data')
def get_audio_data():
    min_freq, max_freq = recorder.get_frequency_range()
    return jsonify({
        "signal_strength": recorder.signal_data[-100:] if recorder.signal_data else [],
        "frequency": recorder.frequency_data[-100:] if recorder.frequency_data else [],
        "freq_range": {
            "min": min_freq,
            "max": max_freq
        }
    })


@app.route('/set_filter', methods=['POST'])
def set_filter():
    data = request.get_json()
    recorder.audio_filter.filter_enabled = data.get('enabled', False)
    recorder.audio_filter.set_filter_strength(data.get('strength', 0.5))
    recorder.audio_filter.set_threshold(data.get('threshold', 0.02))
    return jsonify({"status": "success"})


if __name__ == '__main__':
    app.run(debug=True)