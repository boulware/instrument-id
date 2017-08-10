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
	"reflect"
)

func main() {
	fmt.Println(math.MaxInt32)
	
	wavFileName := "data/trumpet/c5(16).wav"
	wavFile, err := os.Open(wavFileName);
	checkErr(err)
	wavReader, err := wav.New(wavFile)
	checkErr(err)

	fmt.Println(wavReader)
	sampleRate := wavReader.Header.SampleRate
	sampleCount := wavReader.Samples

	wavMagnitudes, err := wavReader.ReadSamples(sampleCount)
	wavMagnitudesInt16 := wavMagnitudes.([]int16)
	wavMagnitudesFloat64 := make([]float64, len(wavMagnitudesInt16))
	for i := range wavMagnitudesInt16 {
		wavMagnitudesFloat64[i] = float64(wavMagnitudesInt16[i])
	}
	
	fmt.Println(reflect.TypeOf(wavMagnitudesInt16))
	checkErr(err)

	// complex values of form (time, magnitude) for use in FFT
	wavComplex := make([]complex128, sampleCount, sampleCount)
	for i := 0; i < sampleCount; i++ {
		wavComplex[i] = complex(float64(i) / float64(sampleRate), wavMagnitudesFloat64[i])
	}

	wavFFT := fft.FFT(wavComplex)
	
	plt, err := plot.New()
	checkErr(err)
	
	wavToXYs := func(wavData []complex128) plotter.XYs {
		gran := 1
		pts := make(plotter.XYs, len(wavData) / gran)
		for i := 0; i < len(wavData) / gran; i += 1 {
			val := wavData[i * gran]
			pts[i].X = real(val)
			pts[i].Y = imag(val)
			if imag(val) > 1e8 {
				fmt.Println(real(val), imag(val))
			}
		}

		return pts
	}

	wavXYs := wavToXYs(wavFFT)
	
	err = plotutil.AddScatters(plt,
		"", wavXYs)
	checkErr(err)

	fmt.Println("generating plot")
	outputFileName := "WaveForm.png"
	err = plt.Save(50*vg.Inch, 4*vg.Inch, outputFileName)
	checkErr(err)	
}

func checkErr(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
