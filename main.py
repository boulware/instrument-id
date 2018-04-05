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

freq = 491
print(FreqToNoteName(freq))

def FindPeak(array):
	pass

np.set_printoptions(threshold=np.inf)

# Creates an immutable tuple to store WAV file data
#WAVData = namedtuple(	'WAVData',
#						['raw_bytes',
#						'sample_width_data',
#						'channel_count',
#						'sample_width',
#						'sampling_frequency',
#						'frame_count',
#						'compression_type'])

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

# Convert a list of bytes to a corresponding list of samples of appropriate width
def SampleWidthDataFromBytes(byte_list, sample_width):
	sample_width_lists = byte_list.reshape(-1, sample_width)

	# 24-bit is the special case here (8 * 3 bits)
	#if(sample_width == 3):
	#	zeroes = [[0] for n in range(len(sample_width_lists))]
	#	sample_width_lists = np.hstack((sample_width_lists, zeroes))

	sample_width_bytes = [bytes(e) for e in sample_width_lists]
	sample_width_ints = [int.from_bytes(e, byteorder='little', signed=True) for e in sample_width_bytes]

	print("sample_width_lists:\n{}".format(sample_width_lists[1000:1010]))
	print("sample_width_bytes:\n{}".format(sample_width_bytes[1000:1010]))
	print("sample_width_ints:\n{}".format(sample_width_ints[1000:1050]))

	return sample_width_ints
	
	#print(sample_width_data[1000:1100])


def OpenWAVFile():
	file_name = filedialog.askopenfilename()
	WAV = wave.open(file_name, 'rb')

	# TODO: This only works with 16-bit WAVs. I need to expand to 24-bit WAVs as well for sure since they are very common.
	file['raw_bytes']	 		= np.fromstring(WAV.readframes(-1), 'Int8') # encode raw byte data as an array of signed 8-bit integers
	file['channel_count'] 		= WAV.getnchannels()
	file['sample_width'] 		= WAV.getsampwidth()
	file['sampling_frequency']	= WAV.getframerate()
	file['frame_count']			= WAV.getnframes()
	file['compression_type']	= WAV.getcomptype()

	WAV.close()

	# Only support 8- through 32-bit WAV files.
	if(file['sample_width'] < 1 or file['sample_width'] > 4):
		print("Audio file sample width is not supported. Only 8- through 32-bit WAV files are supported.")

	file['sample_width_data'] = SampleWidthDataFromBytes(file['raw_bytes'], file['sample_width'])

	fig = Figure(figsize=(5,5), dpi=100)
	waveform_plot = fig.add_subplot(2, 1, 1)
	fft_plot = fig.add_subplot(2, 1, 2)

	print("frame count: {}".format(file['frame_count']))
	k = np.arange(file['frame_count'])
	t = k / file['sampling_frequency'] # Creates discrete array of time values for our sampling frequency
	T = file['frame_count'] / file['sampling_frequency'] # Sample length in seconds
	frq = k / T

	print("channel count={}".format(file['channel_count']))
	print("sample width={}".format(file['sample_width']))
	print("sample frequency={}".format(file['sampling_frequency']))
	print("frame count={}".format(file['frame_count']))

	#print(file['raw_data'])

	global fft_raw_data
	fft_raw_data = scipy.fftpack.fft(file['sample_width_data'])

	waveform_plot.plot(t, file['sample_width_data'])
	fft_plot.plot(frq, fft_raw_data)

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