import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import wave
import numpy as np
from collections import namedtuple
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

import sys

import os.path

import json

from pydub import AudioSegment

from scipy.signal import hilbert

import math

import pyaudio
import binascii
import struct

import sounddevice as sd

#print(sd.query_devices())

import scipy.fftpack

# https://stackoverflow.com/questions/39458337/is-there-a-way-to-add-close-buttons-to-tabs-in-tkinter-ttk-notebook
class CustomNotebook(ttk.Notebook):
    """A ttk Notebook with close buttons on each tab"""

    __initialized = False

    def __init__(self, *args, **kwargs):
        if not self.__initialized:
            self.__initialize_custom_style()
            self.__inititialized = True

        kwargs["style"] = "CustomNotebook"
        ttk.Notebook.__init__(self, *args, **kwargs)

        self._active = None

        self.bind("<ButtonPress-1>", self.on_close_press, True)
        self.bind("<ButtonRelease-1>", self.on_close_release)

    def on_close_press(self, event):
        """Called when the button is pressed over the close button"""

        element = self.identify(event.x, event.y)

        if "close" in element:
            index = self.index("@%d,%d" % (event.x, event.y))
            self.state(['pressed'])
            self._active = index

    def on_close_release(self, event):
        """Called when the button is released over the close button"""
        if not self.instate(['pressed']):
            return

        element =  self.identify(event.x, event.y)
        index = self.index("@%d,%d" % (event.x, event.y))

        if "close" in element and self._active == index:
            self.forget(index)
            self.event_generate("<<NotebookTabClosed>>")

        self.state(["!pressed"])
        self._active = None

    def __initialize_custom_style(self):
        style = ttk.Style()
        self.images = (
            tk.PhotoImage("img_close", data='''
                R0lGODlhCAAIAMIBAAAAADs7O4+Pj9nZ2Ts7Ozs7Ozs7Ozs7OyH+EUNyZWF0ZWQg
                d2l0aCBHSU1QACH5BAEKAAQALAAAAAAIAAgAAAMVGDBEA0qNJyGw7AmxmuaZhWEU
                5kEJADs=
                '''),
            tk.PhotoImage("img_closeactive", data='''
                R0lGODlhCAAIAMIEAAAAAP/SAP/bNNnZ2cbGxsbGxsbGxsbGxiH5BAEKAAQALAAA
                AAAIAAgAAAMVGDBEA0qNJyGw7AmxmuaZhWEU5kEJADs=
                '''),
            tk.PhotoImage("img_closepressed", data='''
                R0lGODlhCAAIAMIEAAAAAOUqKv9mZtnZ2Ts7Ozs7Ozs7Ozs7OyH+EUNyZWF0ZWQg
                d2l0aCBHSU1QACH5BAEKAAQALAAAAAAIAAgAAAMVGDBEA0qNJyGw7AmxmuaZhWEU
                5kEJADs=
            ''')
        )

        style.element_create("close", "image", "img_close",
                            ("active", "pressed", "!disabled", "img_closepressed"),
                            ("active", "!disabled", "img_closeactive"), border=8, sticky='')
        style.layout("CustomNotebook", [("CustomNotebook.client", {"sticky": "nswe"})])
        style.layout("CustomNotebook.Tab", [
            ("CustomNotebook.tab", {
                "sticky": "nswe", 
                "children": [
                    ("CustomNotebook.padding", {
                        "side": "top", 
                        "sticky": "nswe",
                        "children": [
                            ("CustomNotebook.focus", {
                                "side": "top", 
                                "sticky": "nswe",
                                "children": [
                                    ("CustomNotebook.label", {"side": "left", "sticky": ''}),
                                    ("CustomNotebook.close", {"side": "left", "sticky": ''}),
                                ]
                        })
                    ]
                })
            ]
        })
    ])

note_names = ['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B']

def FreqToNoteName(freq):
	A4_freq = 440.0
	A4_midi_number = 69
	C4_midi_number = 60
	notes_per_octave = 12
	midi_number = round(A4_midi_number + notes_per_octave * math.log(freq / A4_freq, 2))
	
	relative_midi_number_C4 = midi_number - C4_midi_number

	note_name = note_names[relative_midi_number_C4 % notes_per_octave]
	note_octave = int(4 + relative_midi_number_C4 / notes_per_octave)

	return '{}{}'.format(note_name, note_octave)

freq = 531
print(FreqToNoteName(freq))

def FindPeak(array):
	pass

np.set_printoptions(threshold=np.inf)

root = tk.Tk()

root.attributes('-zoomed', True)
root.title("Instrument Identification")
tab_control = ttk.Notebook(root)
tab1 = ttk.Frame(tab_control)      
tab_control.add(tab1, text='Tab 1')
tab_control.pack(expand=1, fill="both")

file = {}
fft_raw_data = []

chunk = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 44100

A0_freq = 27.5

class Waveform:
	def RecordFromMicrophone(self):
		p = pyaudio.PyAudio()
		stream = p.open(format				= FORMAT,
						channels			= CHANNELS,
						rate				= SAMPLE_RATE,
						input				= True,
						frames_per_buffer	= chunk)

		mic_data = []
		chunk_count = int(SAMPLE_RATE / chunk * 1.0)

		for i in range(0, chunk_count):
			data = stream.read(chunk)
			mic_data.append(data)

		stream.stop_stream()
		stream.close()
		p.terminate()

		raw_bytes = b''.join(mic_data) # flatten mic_data to a list of bytes

		# This only supports mono atm
		raw_samples = [] # to store mic_data encoded as 16-bit signed integers
		print(str(len(mic_data[2*i:(2*i)+2])))
		for i in range(0, len(mic_data) // 2):
			mic_ints += struct.unpack('<h', mic_data[2*i:(2*i)+2])

		self.channel_count 		= CHANNELS
		self.sample_width 		= 2 # 16 bits
		self.sampling_frequency	= SAMPLE_RATE
		self.frame_count		= len(raw_samples)

		if(self.channel_count == 1):
			self.time_samples = raw_samples		

		self.sample_count = len(self.time_samples)
		self.times = [k / self.sampling_frequency for k in range(self.sample_count)]
		T = self.sample_count / self.sampling_frequency
		self.freqs = [k / T for k in range(self.sample_count)]


		if(self.sample_width < 1 or self.sample_width > 4):
			print("Audio file sample width is not supported. Only 8-, 16-, 24-, and 32-bit WAV and MP3 files are currently supported. Undefined behavior may follow.")

		self.trimmed_time_samples, self.trim_start, self.trim_end = self.Trim()	

	def LoadFromFile(self, file_path):
		self.file_path = file_path

		if file_path[-4:].lower() == '.wav':
			print("Loading .wav file: {}".format(file_path))

			wav_file = wave.open(file_path, 'rb')

			self.channel_count 		= wav_file.getnchannels()
			self.sample_width 		= wav_file.getsampwidth()
			self.sampling_frequency	= wav_file.getframerate()
			self.frame_count		= wav_file.getnframes()

			raw_bytes = np.fromstring(wav_file.readframes(-1), 'Int8') # encode raw byte data as an array of signed 8-bit integers
			raw_samples = SampleWidthDataFromBytes(raw_bytes, self.sample_width)

			wav_file.close()
		elif file_path[-4:].lower() == '.mp3':
			print("Loading .mp3 file: {}".format(file_path))

			mp3_file = AudioSegment.from_file(file_path, format='mp3')

			self.channel_count = mp3_file.channels
			self.sample_width = mp3_file.sample_width
			self.sampling_frequency = mp3_file.frame_rate
			self.frame_count = mp3_file.frame_count

			raw_samples = mp3_file.get_array_of_samples()
		else:
			print("Only .wav and .mp3 files are currently supported.")
			return

		if(self.channel_count == 1):
			self.time_samples = raw_samples
		elif(self.channel_count == 2):
			left_channel = raw_samples[::2]
			right_channel = raw_samples[1::2]
			
			assert(len(left_channel) == len(right_channel))

			self.time_samples = [(left_channel[i] + right_channel[i]) * 0.5 for i in range(len(left_channel))]
		else:
			print("Audio files with more than 2 channels are not supported.")

		self.sample_count = len(self.time_samples)
		self.times = [k / self.sampling_frequency for k in range(self.sample_count)]
		T = self.sample_count / self.sampling_frequency
		self.freqs = [k / T for k in range(self.sample_count)]

		#print("F_s={}".format(self.sampling_frequency))

		if(self.sample_width < 1 or self.sample_width > 4):
			print("Audio file sample width is not supported. Only 8-, 16-, 24-, and 32-bit WAV and MP3 files are currently supported. Undefined behavior may follow.")

		self.trimmed_time_samples, self.trim_start, self.trim_end = self.Trim()

	def GetWaveform(self):
		return self.time_samples, self.times

	def GetSTFT(self):
		f, t, Zxx = scipy.signal.stft(self.trimmed_time_samples, self.sampling_frequency, nperseg=pow(2,16))
		
		averaged_stft = AverageSTFT(Zxx)

		print(len(averaged_stft))

		return averaged_stft, f

	def GetFFT(self):
		#sample_count = len(self.time_samples)

		#k = np.arange(sample_count)
		#t = k / self.sampling_frequency
		#T = sample_count / self.sampling_frequency
		#frq = k / T

		self.freq_samples = abs(scipy.fftpack.rfft(self.time_samples))

		return self.freq_samples, self.freqs

	def GetTrimmedWaveform(self):
		return self.trimmed_time_samples, [x / self.sampling_frequency for x in range(len(self.trimmed_time_samples))]
		
	def GetTrimmedFFT(self):
		#sample_count = len(self.trimmed_time_samples)

		#k = np.arange(sample_count)
		#t = k / self.sampling_frequency
		#T = sample_count / self.sampling_frequency
		#frq = k / T

		self.freq_samples = abs(scipy.fftpack.rfft(self.trimmed_time_samples, self.sample_count))

		return self.freq_samples, self.freqs

	# Returns a slice of the waveform in time domain from start_time (in seconds) to end_time (in seconds)
	def GetTimeSlice(self, start_time, end_time):
		start_k = int(start_time * self.sampling_frequency)
		end_k = int(end_time * self.sampling_frequency)

		k_count = end_k - start_k

		return self.time_samples[start_k:end_k], [start_time + k / self.sampling_frequency for k in range(k_count)]

	#def GetSliceFFTByIndex(self, start_index, end_index):
		

	def GetSliceFFT(self, start_time, end_time):
		time_interval = end_time - start_time
		sample_count = time_interval * self.sampling_frequency

		k = np.arange(sample_count)
		t = start_time + k / self.sampling_frequency
		T = sample_count / self.sampling_frequency
		frq = k / T

		#self.t = self.k / self.sampling_frequency # Creates discrete array of time values for our sampling frequency		
		#self.k = np.arange(self.frame_count)
		#self.T = self.frame_count / self.sampling_frequency # Sample length in seconds
		#self.frq = self.k / self.T

		freq_samples = abs(scipy.fftpack.fft(self.GetTimeSlice(start_time, end_time)))

		#print(freq_samples)

		if len(frq) > len(freq_samples):
			frq = frq[:-1]
		if len(frq) < len(freq_samples):
			freq_samples[:-1]

		return freq_samples[0][:2000], frq[:2000]

	# Find the closest corresponding index in the frequency domain array for the given freq
	def FreqToIndex(self, target_freq):
		index = None
		for i, _ in enumerate(self.freqs[1:]):
			if self.freqs[i-1] < target_freq and self.freqs[i] > target_freq:
				if abs(self.freqs[i-1] - target_freq) < abs(self.freqs[i] - target_freq):
					index = i-1
				else:
					index = i

		if index == None:
			return 0
			print("Tried to convert invalid frequency to index")
		else:
			return index

	def IndexToFreq(self, target_index):
		return self.freqs[target_index]

	# Trim portions of the time domain samples to remove beginning and ending silence as well as maybe some of the attack and release.
	def Trim(self):
		attack_amplitude_threshold = 0.10
		release_amplitude_threshold = 0.10
		anomaly_threshold = 1000
		wave_start_index, wave_end_index = None, None

#		hilbert = np.abs(scipy.signal.hilbert(self.time_samples))
#		b, a = scipy.signal.butter(3, 0.001, btype='lowpass') # 24Hz (for 48k sample rate) 3rd order Butterworth lowpass filter
#		zi = scipy.signal.lfilter_zi(b, a)
#		zi = zi * self.time_samples[0]
#		self.characteristic_signal = scipy.signal.filtfilt(b, a, hilbert)

		rectified_signal = np.abs(self.time_samples)
		b, a = scipy.signal.butter(3, self.FreqToNyquistRatio(100), btype='lowpass') # 24Hz (for 48k sample rate) 3rd order Butterworth lowpass filter
		zi = scipy.signal.lfilter_zi(b, a)
		zi = zi * self.time_samples[0]	
		self.characteristic_signal = scipy.signal.filtfilt(b, a, rectified_signal)

		# First, we find the max amplitude of the characteristic signal
		max_amplitude = np.amax(self.characteristic_signal)
		min_amplitude = np.amin(self.characteristic_signal)

		anomaly_count = 0

		for index, sample in enumerate(self.characteristic_signal):
			if wave_start_index == None:			
				#if (abs(sample) > amplitude_threshold * max_amplitude) or (abs(sample) < amplitude_threshold * min_amplitude):
				if (abs(sample) > attack_amplitude_threshold * max_amplitude):
					anomaly_count += 1
					#print("index={}; abs(sample)={}; anomalies={}".format(index, abs(sample), anomaly_count))
				else:
					anomaly_count = 0

				if anomaly_count >= anomaly_threshold:
					wave_start_index = index - anomaly_threshold + 1
					anomaly_count = 0
			elif wave_end_index == None:
				#if (abs(sample) < amplitude_threshold * max_amplitude) and (abs(sample) > amplitude_threshold * min_amplitude):
				if (abs(sample) < release_amplitude_threshold * max_amplitude):
					anomaly_count += 1
					#print("index={}; abs(sample)={}; anomalies={}".format(index, abs(sample), anomaly_count))
				else:
					anomaly_count = 0

				if anomaly_count >= anomaly_threshold:
					wave_end_index = index - anomaly_threshold + 1
					anomaly_count = 0
					break

		if wave_start_index == None:
			wave_start_index = 0
		if wave_end_index == None:
			wave_end_index = len(self.time_samples)

		print("untrimmed length={}".format(len(self.time_samples)))
		print("wave_start_index={}".format(wave_start_index))
		print("wave_end_index={}".format(wave_end_index))
		print("max_amplitude={}".format(max_amplitude))

		return self.time_samples[wave_start_index:wave_end_index], wave_start_index / self.sampling_frequency, wave_end_index / self.sampling_frequency

	def FreqToNyquistRatio(self, freq):
		nyquist_freq = self.sampling_frequency / 2
		return freq / nyquist_freq

	def FindStableWaveform(self):
		variation_threshold = 0.2
		global_amplitude_threshold = 0.25

		# We probably need at least 10 full wavelengths for now. With a low-end of 100Hz, that gives us 100 waveforms per second => 10 waveforms per 0.1 seconds.
		# So a 0.1 second window should work.
		sample_window_width = int(0.1 * self.sampling_frequency)

		global_max = np.amax(self.characteristic_signal)

		for i in range(len(self.characteristic_signal) - sample_window_width):
			subset = self.characteristic_signal[i:i + sample_window_width - 1]
			max_value = np.amax(subset)
			min_value = np.amin(subset)
			average_value = np.average(subset)

			if average_value >= global_amplitude_threshold * global_max:
				if (max_value - min_value) / average_value <= variation_threshold:
					return self.time_samples[i:i + sample_window_width - 1], self.times[i:i + sample_window_width - 1]

		print("Unable to find stable sub-waveform in waveform.")
		return [], []

	def DetectFreqPeaks(self, rising_threshold = 0.01, falling_threshold = 0.5, peak_amplitude_threshold_ratio = 0.02):
		try:
			self.freq_samples
		except:
			self.GetTrimmedFFT()

		window_width = 1000
		window_std = A0_freq / 2
		window = scipy.signal.gaussian(window_width, window_std)
		self.convolved_fft = scipy.signal.convolve(self.freq_samples, window, mode='same')

		rising_trigger_value = np.max(self.convolved_fft) * rising_threshold
		amplitude_threshold_value = peak_amplitude_threshold_ratio * np.amax(self.freq_samples)

		convolved_slope = np.diff(self.convolved_fft)

		self.peak_freq_indices = []
		self.begin_threshold_indices = []
		self.end_threshold_indices = []

		threshold_activated = False
		convolved_local_max = -1

		for index, convolved_value in enumerate(self.convolved_fft):
			if convolved_value >= rising_trigger_value and not threshold_activated and convolved_slope[index] > 0:			
				start_index = index

				threshold_activated = True				
				self.begin_threshold_indices.append(index)

			if convolved_value <= falling_threshold * convolved_local_max and threshold_activated and convolved_slope[index] < 0:
				end_index = index
				data_local_max = np.max(self.freq_samples[start_index:end_index])
				if data_local_max >= amplitude_threshold_value:
					self.peak_freq_indices.append(start_index + np.argmax(self.freq_samples[start_index:end_index]))

				threshold_activated = False
				self.end_threshold_indices.append(index)
				convolved_local_max = -1

			if threshold_activated:
				if convolved_value > convolved_local_max:
					convolved_local_max = convolved_value

		return self.peak_freq_indices

	def GeneratePlots(self, debug_plots = False):
		try:
			self.freq_samples
		except:
			self.GetTrimmedFFT()

		try:
			self.peak_indices
		except:
			self.DetectFreqPeaks()

		fig, (ax_time, ax_freq) = plt.subplots(2, 1)

		ax_time.plot(self.times, self.time_samples)
		ax_time.set_xlim(xmin = 0, xmax = self.times[-1])	
		if debug_plots == True:
			ax_time.plot(self.times, self.characteristic_signal, color='orange')
			ax_time.axvline(self.trim_start, color='red')
			ax_time.axvline(self.trim_end, color='red')
			ax_time.axhline(color='purple', linewidth=0.8)

		peak_freqs = [self.freqs[i] for i in self.peak_freq_indices]
		peak_amplitudes = [self.freq_samples[i] for i in self.peak_freq_indices]
		max_peak_amplitude = np.amax(self.freq_samples)
		peak_ratios = [k / max_peak_amplitude for k in peak_amplitudes]

		ax_freq.plot(self.freqs, self.freq_samples)
		ax_freq.set_xlim(xmin = 0, xmax = self.freqs[self.peak_freq_indices[-1]] * 1.2)
		ax_freq.set_ylim(ymin = 0, ymax = np.amax(self.freq_samples) * 1.2)

		ax_freq.scatter(peak_freqs, peak_amplitudes, color='purple', marker='o', s=8, zorder=10)
		annotate_font = {'size': 8}
		matplotlib.rc('font', **annotate_font)
		for i, _ in enumerate(peak_freqs):
			ax_freq.annotate("{:.1f}\n({:.2f})".format(peak_freqs[i], peak_ratios[i]), xy=(peak_freqs[i], peak_amplitudes[i]), xytext=(5, 0), textcoords='offset pixels')

		if debug_plots == True:
			ax_freq.plot(self.freqs, self.convolved_fft)

			begin_threshold_freqs = [self.freqs[i] for i in self.begin_threshold_indices]
			end_threshold_freqs = [self.freqs[i] for i in self.end_threshold_indices]		
			begin_threshold_amplitudes = [self.convolved_fft[i] for i in self.begin_threshold_indices]
			end_threshold_amplitudes = [self.convolved_fft[i] for i in self.end_threshold_indices]
			ax_freq.scatter(begin_threshold_freqs, begin_threshold_amplitudes , color='green', marker='^', s=8, zorder=13)
			ax_freq.scatter(end_threshold_freqs, end_threshold_amplitudes, color='red', marker='v', s=8, zorder=13)			

		fig.show()	

def RecordMic():
	sound = Waveform()
	sound.RecordMic()

	fig, (ax_time, ax_freq) = plt.subplots(2, 1)

	fig = Figure(figsize=(5,5), dpi=100)
	waveform_plot = fig.add_subplot(2, 1, 1)
	fft_plot = fig.add_subplot(2, 1, 2)

	n = len(mic_ints) # number of samples
	k = np.arange(n)
	t = k / RATE # Creates discrete array of time values for our sampling frequency
	T = n / RATE # Sample length in seconds
	frq = k / T
	frq = frq[range(n//2)]
	#frq = np.linspace(0.0, 1.0/(2.0*T), N/2)



	global fft_raw_data
	fft_raw_data = scipy.fftpack.fft(mic_ints)
	fft_raw_data = fft_raw_data[range(n//2)]

	print(str(len(t)))
	print(str(len(mic_ints)))
	waveform_plot.plot(t, mic_ints)
	fft_plot.plot(frq, fft_raw_data)

	canvas = FigureCanvasTkAgg(fig, master=root)
	canvas.show()
	canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

	toolbar = NavigationToolbar2TkAgg(canvas, root)
	toolbar.update()
	canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Convert a list of bytes to a corresponding list of samples (as signed ints) of appropriate byte width.
def SampleWidthDataFromBytes(byte_list, sample_width):
	sample_width_lists = byte_list.reshape(-1, sample_width)

	sample_width_bytes = [bytes(e) for e in sample_width_lists]
	sample_width_ints = [int.from_bytes(e, byteorder='little', signed=True) for e in sample_width_bytes]

	return sample_width_ints
	
def AverageSTFT(data, to_print=False):
	data_sum = np.sum(data, 1)
	data_average = np.divide(data_sum, len(data))
	data_magnitude = np.absolute(data_average)

	if to_print:
		print("data_sum:{}".format(data_sum))
		print("data_average:{}".format(data_average))
		print("data_magnitude:{}".format(data_magnitude))

	return data_magnitude


def DetectLocalPeaks(data):
	min_indices = [0, len(data)-1]
	max_indices = []

	for i in np.arange(1, len(data)-2):
		if data[i] < data[i] and data[i] < data[i+2]:
			min_indices.append(i)		
		elif data[i] > data[i-1] and data[i] > data[i+1]:
			max_indices.append(i)

	np.sort(min_indices)

	return min_indices, max_indices

def ThresholdFilterMaxima(convolved_data, threshold_value):
	min_indices, max_indices = DetectLocalPeaks(convolved_data)

def DistanceFilter(data, peak_indices, minimum_peak_distance):
	filtered_peak_indices = []
	cluster_start_index = 0
	cluster_peak_index = 0

	print("mpd={}".format(minimum_peak_distance))
	print("peaks remaining={}".format(len(peak_indices)))
	print(peak_indices)

	for i, peak_index in enumerate(peak_indices):
		if i == 0:
			cluster_start_index = peak_index
			cluster_peak_index = peak_index
			continue

		if np.abs(peak_index - cluster_start_index) < minimum_peak_distance:
			print("peak_index={}; cluster_start_index={}".format(peak_index, cluster_start_index))
			if data[peak_index] > data[cluster_peak_index]:
					cluster_peak_index = peak_index
		else:
			filtered_peak_indices.append(cluster_peak_index)			
			cluster_start_index = peak_index
			cluster_peak_index = peak_index

		if i == len(peak_indices)-1 and cluster_start_index == peak_index:
			filtered_peak_indices.append(cluster_peak_index)

	if len(peak_indices) != len(filtered_peak_indices):
		return DistanceFilter(data, filtered_peak_indices, minimum_peak_distance)
	else:
		return filtered_peak_indices



def oldDetectPeaks(data, convolved_data, start_index, peak_threshold, minimum_peak_height, convolved_minimum_peak_height, minimum_peak_distance):
	# We place peaks at the beginning and end of the spectrum
	peak_indices = [] # All local mins or maxes, regardless of whether they meet the criteria
	peak_indices.append(0)
	peak_indices.append(len(data) - 1)

	for i in np.arange(1, len(data) - 2):
		if np.sign(data[i] - data[i-1]) == np.sign(data[i] - data[i+1]):
			peak_indices.append(i)

	peak_indices = np.sort(peak_indices)

	filtered_peak_indices = []
	peak_groups = zip(peak_indices, peak_indices[1:], peak_indices[2:])
	for peak_group in peak_groups:
		if (data[peak_group[1]] / data[peak_group[0]] > peak_threshold and data[peak_group[1]] / data[peak_group[2]] > peak_threshold):
			if data[peak_group[1]] > minimum_peak_height:
				if np.max(data[peak_group[0]:peak_group[2]]) > minimum_peak_height:
					if np.max(convolved_data[peak_group[0]:peak_group[2]]) > convolved_minimum_peak_height:
						filtered_peak_indices.append(peak_group[1])


	distance_filtered_peak_indices = DistanceFilter(data, filtered_peak_indices, minimum_peak_distance)

	return peak_indices, filtered_peak_indices, distance_filtered_peak_indices





def OpenWAVFile(file_path = None):
	sound = Waveform()

	if not file_path:
		file_path = filedialog.askopenfilename()

	sound.LoadFromFile(file_path)

	fig, (ax_time, ax_freq) = plt.subplots(2, 1)

	#values, freqs = sound.GetTrimmedFFT()

	sound.GeneratePlots()

	#sound.DetectFreqPeaks(0.01, 0.5, 0.05)

	#peak_freqs = [sound.freqs[i] for i in peak_indices]
	#begin_threshold_freqs = [sound.freqs[i] for i in begin_threshold_indices]
	#end_threshold_freqs = [sound.freqs[i] for i in end_threshold_indices]
#
#	#peak_values = [values[i] for i in peak_indices]
#	#begin_threshold_values = [convolved_fft[i] for i in begin_threshold_indices]
#	#end_threshold_values = [convolved_fft[i] for i in end_threshold_indices]
#
#	#max_peak_amplitude = np.amax(values)
#	#peak_ratios = [k / max_peak_amplitude for k in peak_values]
#	#ax_freq.scatter(peak_freqs, peak_values, color='purple', marker='o', s=8, zorder=10)
#	#annotate_font = {'size': 8}
#	#matplotlib.rc('font', **annotate_font)
#	#for i, peak_freq in enumerate(peak_freqs):
#	#	ax_freq.annotate("{:.1f}\n({:.2f})".format(peak_freq, peak_ratios[i]), xy=(peak_freq, peak_values[i]), xytext=(5, 0), textcoords='offset pixels')
#	##freq_plot.scatter(filtered_peak_freqs, filtered_peak_values, color='blue', marker='o', s=8, zorder=11)
#	##freq_plot.scatter(distance_filtered_peak_freqs, distance_filtered_peak_values, color='white', marker='o', s=4, zorder=12)
#	#debug_plots = False
#	#if debug_plots == True:
#	#	ax_freq.plot(freqs, convolved_fft)	
#	#	ax_freq.scatter(begin_threshold_freqs, begin_threshold_values, color='green', marker='^', s=8, zorder=13)
#	#	ax_freq.scatter(end_threshold_freqs, end_threshold_values, color='red', marker='v', s=8, zorder=13)	
#
#	##ax_time.subplot(2, 1, 1)
#	#ax_time.plot(sound.times, sound.time_samples)
#	#ax_time.set_xlim(xmin = 0, xmax = sound.times[-1])	
#	#if debug_plots == True:
#	#	ax_time.plot(sound.times, sound.characteristic_signal, color='orange')
#	#	ax_time.axvline(sound.trim_start, color='red')
#	#	ax_time.axvline(sound.trim_end, color='red')
#	#	ax_time.axhline(color='purple', linewidth=0.8)
#
#	##plt.subplot(2, 1, 2)
#	#ax_freq.plot(freqs, values, color='grey')
#	#ax_freq.set_xlim(xmin = 0, xmax = sound.sampling_frequency//2)
#	#ax_freq.set_ylim(ymin = 0, ymax = np.amax(values) * 1.2)
	#ax_freq.axhline(color='purple', linewidth=0.8)

	#plt.show()

	#toolbar = NavigationToolbar2TkAgg(canvas, root)
	#toolbar.update()

def ExportCSV():
	csv_filename = filedialog.asksaveasfilename(filetypes=[("Comma-separated values", 'csv')], defaultextension='csv')
	with open(csv_filename, 'wt') as csv_file:
		global fft_raw_data
		for e in fft_raw_data:
			csv_file.write(str(e))
			csv_file.write(',\n')
			#csv_file.write(np.array_str(fft_raw_data))

menubar = tk.Menu(root)
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label="Open...", command=OpenWAVFile, underline=0)
filemenu.add_cascade(label="Export...", command=ExportCSV, underline=0)
filemenu.add_command(label="Record", command=RecordMic, underline=0)
menubar.add_cascade(label="File", menu=filemenu)
root.config(menu=menubar)

if len(sys.argv) == 2:
	OpenWAVFile(sys.argv[1])

root.mainloop()