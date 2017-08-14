package main

import (
	"fmt"
	"log"
	"os"


	"github.com/gonum/plot"
	"github.com/gonum/plot/plotter"
	"github.com/gonum/plot/plotutil"
	"github.com/gonum/plot/vg"

	"github.com/mjibson/go-dsp/wav"	
	"github.com/mjibson/go-dsp/fft"

	"math"
//	"reflect"
)

func main() {
	//	wavFileName := "data/trumpet/c5(16).wav"
	wavFileName := "data/sin432(16bit).wav"
	
	wavFile, err := os.Open(wavFileName);
	checkErr(err)
	wavReader, err := wav.New(wavFile)
	checkErr(err)

	sampleRate := wavReader.Header.SampleRate
	channelCount := wavReader.NumChannels
	sampleCount := wavReader.Samples // number of data samples
	sampleLength := sampleCount / int(channelCount) // # of data samples normalized by channel count

	fmt.Println("sample count:", sampleCount)
	fmt.Println("sample length:", sampleLength)

	wavMagnitudes, err := wavReader.ReadSamples(sampleCount)
	wavMagnitudesInt16 := wavMagnitudes.([]int16)
	wavMagnitudesFloat64 := make([]float64, sampleLength)
	if channelCount == 1 {
		for i := range wavMagnitudesInt16 {
			wavMagnitudesFloat64[i] = float64(wavMagnitudesInt16[i])
		}
	} else if channelCount == 2 {
		for i := 0; i < sampleCount; i += 2 {
			wavMagnitudesFloat64[i/2] = 0.5 * (float64(wavMagnitudesInt16[i]) + float64(wavMagnitudesInt16[i+1]))
		}
	}

	
	wavTimeMagnitudeTable := make([]complex128, sampleLength, sampleLength)
	for i := 0; i < sampleLength; i++ {
		wavTimeMagnitudeTable[i] = complex(float64(i) / float64(sampleRate), wavMagnitudesFloat64[i])
	}

	wavFFT := fft.FFT(wavTimeMagnitudeTable)
	for ind, val := range wavFFT {
		wavFFT[ind] = val / complex(float64(sampleLength), 0.0)
	}
	
	plt, err := plot.New()
	checkErr(err)
	
	wavToXYs := func(wavData []complex128) plotter.XYs {
		gran := 1
		pts := make(plotter.XYs, len(wavData) / gran)
		for i := 0; i < len(wavData) / gran; i += 1 {
			val := wavData[i]
			pts[i].X = real(val)
			pts[i].Y = imag(val)
		}

		return pts
	}

	
	wavFreqAmpTable := make([]complex128, sampleLength, sampleLength)
	for i, val := range wavFFT {
		x := real(val)
		y := imag(val)
		frequency := float64(i) * (float64(sampleRate) / float64(sampleLength))
		amplitude := math.Sqrt(math.Pow(x, 2.0) + math.Pow(y, 2.0))
		wavFreqAmpTable[i] = complex(frequency, amplitude)
		if amplitude > 200 {
			fmt.Println(frequency, amplitude)
		}
	}

//	wavXYs := wavToXYs(wavFreqAmpTable[0 : len(wavFreqAmpTable) / 2])
	wavXYs := wavToXYs(wavTimeMagnitudeTable)
	
/*	err = plotutil.AddLinePoints(plt,
		"", wavXYs)
	checkErr(err)*/

	err = plotutil.AddScatters(plt,
		"", wavXYs)
	checkErr(err)

	fmt.Println("generating plot")
	outputFileName := "WaveForm.png"
	err = plt.Save(100*vg.Inch, 10*vg.Inch, outputFileName)
	checkErr(err)	
}

func checkErr(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
