import tkinter as tk
from tkinter import filedialog
import wave
import numpy as np
from collections import namedtuple
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

import detect_peaks

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
file = {}
fft_raw_data = []

chunk = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

def RecordMic():
	p = pyaudio.PyAudio()
	stream = p.open(format				= FORMAT,
					channels			= CHANNELS,
					rate				= RATE,
					input				= True,
					frames_per_buffer	= chunk)

	mic_data = []
	chunk_count = int(RATE / chunk * 1.0)

	for i in range(0, chunk_count):
		data = stream.read(chunk)
		mic_data.append(data)

	stream.stop_stream()
	stream.close()
	p.terminate()

	mic_data = b''.join(mic_data) # flatten mic_data to a list of bytes

	mic_ints = [] # to store mic_data encoded as 16-bit signed integers
	print(str(len(mic_data[2*i:(2*i)+2])))
	for i in range(0, len(mic_data) // 2):
		mic_ints += struct.unpack('<h', mic_data[2*i:(2*i)+2])

	waveFile = wave.open("_file.wav", 'wb')
	waveFile.setnchannels(CHANNELS)
	waveFile.setsampwidth(p.get_sample_size(FORMAT))
	waveFile.setframerate(RATE)
	waveFile.writeframes(mic_data)
	waveFile.close()



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

class Waveform:
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
			self.time_domain_samples = raw_samples
		elif(self.channel_count == 2):
			left_channel = raw_samples[::2]
			right_channel = raw_samples[1::2]
			
			assert(len(left_channel) == len(right_channel))

			self.time_domain_samples = [(left_channel[i] + right_channel[i]) * 0.5 for i in range(len(left_channel))]
		else:
			print("Audio files with more than 2 channels are not supported.")

		self.times = [k / self.sampling_frequency for k in range(len(self.time_domain_samples))]

		if(self.sample_width < 1 or self.sample_width > 4):
			print("Audio file sample width is not supported. Only 8-, 16-, 24-, and 32-bit WAV and MP3 files are currently supported. Undefined behavior may follow.")

		self.trimmed_time_domain_samples, self.trim_start, self.trim_end = self.Trim()



	def GetWaveform(self):
		return self.time_domain_samples, self.times

	def GetSTFT(self):
		f, t, Zxx = scipy.signal.stft(self.trimmed_time_domain_samples, self.sampling_frequency, nperseg=pow(2,16))
		

#		test_array = np.array([
#			[1+1j, 2+2j, 3+3j],
#			[4+4j, 5+5j, 6+6j],
#			[10+10j, 11+11j, 12+12j],
#			[1+1j, 2+2j, 3+3j]
#			[1, 2],
#			[2, 3],
#			[5, 6]
#		])
#
#		test_average = AverageSTFT(test_array, True)

		#print(len(f))
		#print(len(t))
		#print(len(Zxx))
		
		averaged_stft = AverageSTFT(Zxx)

		print(len(averaged_stft))

		return averaged_stft, f

	def GetFFT(self):
		sample_count = len(self.time_domain_samples)

		k = np.arange(sample_count)
		t = k / self.sampling_frequency
		T = sample_count / self.sampling_frequency
		frq = k / T

		freq_domain_samples = abs(scipy.fftpack.rfft(self.time_domain_samples))

		return freq_domain_samples, frq

	def GetTrimmedWaveform(self):
		return self.trimmed_time_domain_samples, [x / self.sampling_frequency for x in range(len(self.trimmed_time_domain_samples))]
		
	def GetTrimmedFFT(self):
		sample_count = len(self.trimmed_time_domain_samples)

		k = np.arange(sample_count)
		t = k / self.sampling_frequency
		T = sample_count / self.sampling_frequency
		frq = k / T

		freq_domain_samples = abs(scipy.fftpack.fft(self.trimmed_time_domain_samples, sample_count))

#		if len(frq) > len(freq_domain_samples):
#			frq = frq[:-1]
#		if len(frq) < len(freq_domain_samples):
#			freq_domain_samples[:-1]


		#return freq_domain_samples[:2000], frq[:2000]
		return freq_domain_samples, frq

	# Returns a slice of the waveform in time domain from start_time (in seconds) to end_time (in seconds)
	def GetTimeSlice(self, start_time, end_time):
		start_k = int(start_time * self.sampling_frequency)
		end_k = int(end_time * self.sampling_frequency)

		k_count = end_k - start_k

		return self.time_domain_samples[start_k:end_k], [start_time + k / self.sampling_frequency for k in range(k_count)]

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

		freq_domain_samples = abs(scipy.fftpack.fft(self.GetTimeSlice(start_time, end_time)))

		#print(freq_domain_samples)

		if len(frq) > len(freq_domain_samples):
			frq = frq[:-1]
		if len(frq) < len(freq_domain_samples):
			freq_domain_samples[:-1]

		return freq_domain_samples[0][:2000], frq[:2000]


	# Trim portions of the time domain samples to remove beginning and ending silence as well as maybe some of the attack and release.
	def Trim(self):
		attack_amplitude_threshold = 0.10
		release_amplitude_threshold = 0.10
		anomaly_threshold = 1000
		wave_start_index, wave_end_index = None, None

#		hilbert = np.abs(scipy.signal.hilbert(self.time_domain_samples))
#		b, a = scipy.signal.butter(3, 0.001, btype='lowpass') # 24Hz (for 48k sample rate) 3rd order Butterworth lowpass filter
#		zi = scipy.signal.lfilter_zi(b, a)
#		zi = zi * self.time_domain_samples[0]
#		self.characteristic_signal = scipy.signal.filtfilt(b, a, hilbert)

		rectified_signal = np.abs(self.time_domain_samples)
		b, a = scipy.signal.butter(3, 0.001, btype='lowpass') # 24Hz (for 48k sample rate) 3rd order Butterworth lowpass filter
		zi = scipy.signal.lfilter_zi(b, a)
		zi = zi * self.time_domain_samples[0]	
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
			wave_end_index = len(self.time_domain_samples)

		print("untrimmed length={}".format(len(self.time_domain_samples)))
		print("wave_start_index={}".format(wave_start_index))
		print("wave_end_index={}".format(wave_end_index))
		print("max_amplitude={}".format(max_amplitude))

		return self.time_domain_samples[wave_start_index:wave_end_index], wave_start_index / self.sampling_frequency, wave_end_index / self.sampling_frequency

	def FindStableWaveform(self):
		variation_threshold = 0.2
		global_amplitude_threshold = 0.75

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
					return self.time_domain_samples[i:i + sample_window_width - 1], self.times[i:i + sample_window_width - 1]

	def FindPeaks(self):
		pass

	def CacheCalculations(self):
		pass


def OpenWAVFile():
	sound = Waveform()

	file_path = filedialog.askopenfilename()
	sound.LoadFromFile(file_path)

	fig = Figure(figsize=(5,5), dpi=100)
	time_plot = fig.add_subplot(2, 1, 1)
	freq_plot = fig.add_subplot(2, 1, 2)

	#sound.GetTimeSlice(1.1, 1.11)
	#print(sound.GetTimeSlice(1.1, 1.11))

	s_t = 0
	e_t = 0.5

	#values, times = sound.GetTrimmedWaveform()
	#time_plot.plot(times, values)

	#values, times = sound.GetWaveform()
	values, times = sound.FindStableWaveform()
	time_plot.plot(times, values)
	#time_plot.plot(times, sound.characteristic_signal, color='orange')

	#time_plot.axvline(sound.trim_start, color='red')
	#time_plot.axvline(sound.trim_end, color='red')

	time_plot.axhline(color='purple', linewidth=0.8)


	values, freqs = sound.GetFFT()

	d_freq = freqs[1] - freqs[0]

	window_width = 1000
	window_std = 10
	window = scipy.signal.gaussian(window_width, window_std)
	convolved_fft = scipy.signal.convolve(values, window, mode='same')

	d1_convolution = np.gradient(convolved_fft, d_freq)

#	freq_plot.plot(freqs, d1_convolution, color='red')

	d2_threshold = -1e6

	#for e in convolved_fft:

	#scipy.optimize.newton(d1_convolution, )
	print("len(convolved_fft):{}".format(len(convolved_fft)))
	peaks = scipy.signal.find_peaks_cwt(convolved_fft, np.arange(10,11))
	#for e in 

	minimum_peak_height = 10e8

	peaks = detect_peaks.detect_peaks(convolved_fft, mph=10e8, mpd=1000)

	peak_freqs = [freqs[peak] for peak in peaks]
	peak_values = [convolved_fft[peak] for peak in peaks]


	freq_plot.axhline(color='purple', linewidth=0.8)

	#values, freqs = sound.GetSTFT()
	freq_plot.plot(freqs, convolved_fft)
	freq_plot.plot(freqs, values, color='green')

	freq_plot.scatter(peak_freqs, peak_values, color='red', marker='o', s=10)

	#waveform_plot.plot(sound.GetTimeSlice(1.1, 1.2))
	#fft_plot.plot(sound.frq, sound.freq_domain_samples)

	canvas = FigureCanvasTkAgg(fig, master=root)
	canvas.show()
	canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

	toolbar = NavigationToolbar2TkAgg(canvas, root)
	toolbar.update()
	canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

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
filemenu.add_cascade(label="Record", command=RecordMic, underline=0)
menubar.add_cascade(label="File", menu=filemenu)
root.config(menu=menubar)

root.mainloop()