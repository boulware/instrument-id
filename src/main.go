package main

import (
	"fmt"
	"log"
	"os"
	
	"github.com/cryptix/wav"

	"github.com/gonum/plot"
	"github.com/gonum/plot/plotter"
	"github.com/gonum/plot/plotutil"
	"github.com/gonum/plot/vg"

	"github.com/mjibson/go-dsp/fft"

	"math"
)

func main() {
	fmt.Println(math.MaxInt32)
	
	//	wavFileName := "data/trumpet/a#3.wav"
	wavFileName := "data/trumpet/c5(16).wav"
	wavFileInfo, err := os.Stat(wavFileName)
	checkErr(err)
	wavFile, err := os.Open(wavFileName)
	checkErr(err)
	wavReader, err := wav.NewReader(wavFile, wavFileInfo.Size())
	checkErr(err)

	fmt.Println(wavReader)
	sampleRate := wavReader.GetFile().SampleRate
	sampleCount := wavReader.GetSampleCount()

//	fmt.Println(wavReader.ReadRawSample())
	
	wavMagnitudes := make([]float64, sampleCount, sampleCount)
	for i := uint32(0); i < sampleCount; i++ {
		value, err := wavReader.ReadSample()
		checkErr(err)

		//		fmt.Printf("%v, %08b => %016b\n", i, value, int16(value[0]) + int16(value[1])<<8)
//		fmt.Printf("%v, %v => %v\n", i, value, int16(value[0]) + int16(value[1])<<8)
		//		wavMagnitudes[i] = float64(int16(value))
		wavMagnitudes[i] = float64(int16(value))
	}

/*	for i := 0; i < 10; i++ {
		fmt.Println(wavMagnitudes[i], ", ", wavMagnitudesInt[i])
	}
*/
//	fmt.Println(wavMagnitudes)
	
	// complex values of form (time, magnitude) for use in FFT
	wavComplex := make([]complex128, sampleCount, sampleCount)
	for i := uint32(0); i < sampleCount; i++ {
		wavComplex[i] = complex(float64(i) / float64(sampleRate), wavMagnitudes[i])
	}

	wavFFT := fft.FFT(wavComplex)
	
	plt, err := plot.New()
	checkErr(err)
	
	wavToXYs := func(wavFFT []complex128) plotter.XYs {
		gran := 1
		pts := make(plotter.XYs, len(wavFFT) / gran)
		for i := 0; i < len(wavFFT) / gran; i += 1 {
			val := wavFFT[i * gran]
//			fmt.Println(imag(val))
			pts[i].X = real(val)
			pts[i].Y = imag(val)
			if imag(val) > 1e8 {
				fmt.Println(real(val), imag(val))
			}
		}

		return pts
	}

	wavXYs := wavToXYs(wavFFT)
/*	for i := 0; i < 100; i++ {
		fmt.Println(wavXYs[i])
	}*/

	
	
	err = plotutil.AddScatters(plt,
		"", wavXYs)
	checkErr(err)

	fmt.Println("generating plot")
	outputFileName := "WaveForm.png"
	err = plt.Save(50*vg.Inch, 4*vg.Inch, outputFileName)
	checkErr(err)

	

	
//	fmt.Println(wav_reader.ReadSample())
//	fmt.Println(wav_reader.ReadSampleEvery(1000, 0))
//	fmt.Println(wav_reader.ReadSampleEvery(1000, 0))	
}

func checkErr(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
