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
	

class Waveform:
	def LoadFromFile(self, file_path):
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


		if(self.sample_width < 1 or self.sample_width > 4):
			print("Audio file sample width is not supported. Only 8-, 16-, 24-, and 32-bit WAV files are currently supported. Undefined behavior may follow.")

		self.trimmed_time_domain_samples, self.trim_start, self.trim_end = self.Trim()


	def GetWaveform(self):
		return self.time_domain_samples, [k / self.sampling_frequency for k in range(len(self.time_domain_samples))]

	def GetFFT(self):
		sample_count = len(self.time_domain_samples)

		k = np.arange(sample_count)
		t = k / self.sampling_frequency
		T = sample_count / self.sampling_frequency
		frq = k / T

		#self.t = self.k / self.sampling_frequency # Creates discrete array of time values for our sampling frequency		
		#self.k = np.arange(self.frame_count)
		#self.T = self.frame_count / self.sampling_frequency # Sample length in seconds
		#self.frq = self.k / self.T

		freq_domain_samples = abs(scipy.fftpack.fft(self.time_domain_samples))

		#print(freq_domain_samples)

#		if len(frq) > len(freq_domain_samples):
#			frq = frq[:-1]
#		if len(frq) < len(freq_domain_samples):
#			freq_domain_samples[:-1]

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
		attack_amplitude_threshold = 0.05
		release_amplitude_threshold = 0.02
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
		max_amplitude = max(self.characteristic_signal)
		min_amplitude = min(self.characteristic_signal)

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

		#anomaly_count = 0

		#time_domain_d = np.gradient(self.time_domain_samples)
		#time_domain_dd = np.gradient(time_domain_d)

		#print("max_amplitude={}".format(max_amplitude))
		#print(self.time_domain_samples[:20])
		#print(time_domain_d[:20])
		#print(time_domain_dd[:20])

#		for index, sample in enumerate(self.time_domain_samples):
#			if wave_start_index == None:			
#				if (sample > amplitude_threshold * max_amplitude) or (sample < amplitude_threshold * min_amplitude):
#					anomaly_count += 1
#					#print("index={}; sample={}; anomalies={}".format(index, sample, anomaly_count))
#				else:
#					anomaly_count = 0
#
#				if anomaly_count >= anomaly_threshold:
#					wave_start_index = index - anomaly_threshold
#					anomaly_count = 0
#			elif wave_end_index == None:
#				if (sample < amplitude_threshold * max_amplitude) and (sample > amplitude_threshold * min_amplitude):
#					anomaly_count += 1
#					#print("index={}; sample={}; anomalies={}".format(index, sample, anomaly_count))
#				else:
#					anomaly_count = 0
#
#				if anomaly_count >= anomaly_threshold:
#					wave_end_index = index - anomaly_threshold
#					anomaly_count = 0
#					break

		# Now that the silence is trimmed, let's take a small clip from the center of the primary waveform to eliminate effects from attack, decay, and release.
		# We probably want at least 100ms of sound, but let's start a bit higher.

		#sample_count = wave_end_index - wave_start_index
		#target_duration = 0.250 # aim for 250ms to start
		#duration = (wave_end_index - wave_start_index) / self.sampling_frequency # in seconds
		#print("duration of trimmed clip: {}".format(duration))
		#print("start time: {}; end time: {}".format(wave_start_index / self.sampling_frequency, wave_end_index / self.sampling_frequency))

		#if duration < target_duration:
		#	print("Trimmed duration may be too short for accurate results. Continuing anyway.")

		#target_sample_count = target_duration * self.sampling_frequency # The number of samples to get in order to have a sample of target_duration

		#offset = int((sample_count - target_sample_count) / 2) # the amount to offset from each side to get a waveform of length target_duration from the center of the trimmed waveform



		return self.time_domain_samples[wave_start_index:wave_end_index], wave_start_index / self.sampling_frequency, wave_end_index / self.sampling_frequency




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

	values, times = sound.GetWaveform()
	time_plot.plot(times, values)
	time_plot.plot(times, sound.characteristic_signal, color='orange')

	time_plot.axvline(sound.trim_start, color='red')
	time_plot.axvline(sound.trim_end, color='red')

	values, freqs = sound.GetFFT()
	freq_plot.plot(freqs, values)

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