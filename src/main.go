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
	sampleCount := wavReader.Samples

	fmt.Println("sample count:", sampleCount)

	wavMagnitudes, err := wavReader.ReadSamples(sampleCount)
	wavMagnitudesInt16 := wavMagnitudes.([]int16)
	wavMagnitudesFloat64 := make([]float64, len(wavMagnitudesInt16))
	for i := range wavMagnitudesInt16 {
		wavMagnitudesFloat64[i] = float64(wavMagnitudesInt16[i])
	}
	
	wavTimeMagnitudeTable := make([]complex128, sampleCount, sampleCount)
	for i := 0; i < sampleCount; i++ {
		wavTimeMagnitudeTable[i] = complex(float64(i) / float64(sampleRate), wavMagnitudesFloat64[i])
	}

	wavFFT := fft.FFT(wavTimeMagnitudeTable)
	for ind, val := range wavFFT {
		wavFFT[ind] = val / complex(float64(sampleCount), 0.0)
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

	
	wavFreqAmpTable := make([]complex128, sampleCount, sampleCount)
	for i, val := range wavFFT {
		x := real(val)
		y := imag(val)
		frequency := float64(i) * (float64(sampleRate) / float64(sampleCount))
		amplitude := math.Sqrt(math.Pow(x, 2.0) + math.Pow(y, 2.0))
		wavFreqAmpTable[i] = complex(frequency, amplitude)
		//		if amplitude > 10 { fmt.Println(frequency, amplitude) }
		if amplitude > 200{
			fmt.Println(frequency, amplitude)
		}
	}

	wavXYs := wavToXYs(wavFreqAmpTable[0 : len(wavFreqAmpTable) / 2])
	
	err = plotutil.AddLinePoints(plt,
		"", wavXYs)
	checkErr(err)

	fmt.Println("generating plot")
	outputFileName := "WaveForm.png"
	err = plt.Save(10*vg.Inch, 10*vg.Inch, outputFileName)
	checkErr(err)	
}

func checkErr(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
