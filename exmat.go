package exmat

import (
	"bytes"
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"sync"
	"time"

	"github.com/gonum/matrix/mat64"
)

// PoolingMode is flag for pooling.
type PoolingMode int

const (
	// Max pooling
	Max = iota + 1
	// Avg shows average pooling.
	Avg
)

// ExMat is extented Matrix of gonum/matrix
type ExMat struct {
	*mat64.Dense
}

// NewExMat is constructor
func NewExMat(r, c int, src []float64) *ExMat {
	return &ExMat{
		mat64.NewDense(r, c, src),
	}
}

// NewExMatFromDense make ExMat from mat64.Dense
func NewExMatFromDense(dense *mat64.Dense) *ExMat {
	return &ExMat{
		dense,
	}
}

// Random generates random ExMat.
func Random(r, c int) *ExMat {
	rand.Seed(time.Now().UnixNano())
	res := make([]float64, r*c)
	for y := 0; y < int(r); y++ {
		for x := 0; x < int(c); x++ {
			res[y*c+x] = rand.Float64()
		}
	}
	return NewExMat(r, c, res)
}

// Zeros generates zero matrix.
func Zeros(r, c int) *ExMat {
	rand.Seed(time.Now().UnixNano())
	res := make([]float64, r*c)
	for y := 0; y < r; y++ {
		for x := 0; x < c; x++ {
			res[y*c+x] = 0.0
		}
	}
	return NewExMat(r, c, res)
}

// GetRow returns the number of rows.
func (emat *ExMat) GetRow() int {
	r, _ := emat.Dims()
	return r
}

// GetCol returns the number of cols.
func (emat *ExMat) GetCol() int {
	_, c := emat.Dims()
	return c
}

func (emat *ExMat) String() string {
	var w bytes.Buffer
	r, c := emat.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			w.WriteString(strconv.FormatFloat(emat.At(i, j), 'f', 4, 64))
			w.WriteString(" ")
		}
		w.WriteString("\n")
	}
	return w.String()
}

func cmp(a, b []float64, resCh chan bool) {
	for i := range a {
		if a[i] != b[i] {
			resCh <- false
		}
	}
	resCh <- true
}

// Equals compares ExMat.
func (emat *ExMat) Equals(t *ExMat) bool {
	rs, cs := emat.Dims()
	rt, ct := t.Dims()
	if rs != rt || cs != ct {
		return false
	}
	ch := make(chan bool, rs)
	for y := 0; y < rs; y++ {
		go cmp(emat.RawRowView(y), t.RawRowView(y), ch)
	}
	for y := 0; y < rs; y++ {
		if res := <-ch; !res {
			return false
		}
	}
	close(ch)
	return true
}

// Reshape reshapes matrix
func (emat *ExMat) Reshape(r, c int, src *ExMat) (err error) {
	rs, cs := src.Dims()
	if rs*cs != r*c {
		return fmt.Errorf("Size is not same")
	}
	row := 0
	col := 0
	newMat := make([]float64, r*c)
	for y := 0; y < r; y++ {
		for x := 0; x < c; x++ {
			if col > cs-1 {
				row++
				col = 0
			}
			newMat[y*c+x] = src.At(row, col)
			col++
		}
	}
	emat.Dense = mat64.NewDense(r, c, newMat)
	return
}

func maxPool(m mat64.Matrix) float64 {
	return mat64.Max(m)
}

func avgPool(m mat64.Matrix) float64 {
	r, c := m.Dims()
	return mat64.Sum(m) / float64(r*c)
}

func (emat *ExMat) execPool(newMat []float64, y, rows, cols, k, s int, mode PoolingMode, wg *sync.WaitGroup) {
	for x := 0; x < cols; x++ {
		part := emat.View(y*s, x*s, k, k)
		switch mode {
		case Max:
			newMat[y*cols+x] = maxPool(part)
		case Avg:
			newMat[y*cols+x] = avgPool(part)
		}
	}
	wg.Done()
}

// Pooling pool the matrix
func (emat *ExMat) Pooling(k, s int, mode PoolingMode, src *ExMat) {
	rs, cs := src.Dims()
	rows := int(math.Ceil(float64(rs-k)/float64(s))) + 1
	cols := int(math.Ceil(float64(cs-k)/float64(s))) + 1
	if (rs-k)%s != 0 {
		padded := src.Grow(1, 1)
		src.Dense = padded.(*mat64.Dense)
		fmt.Println(src.String())
	}
	newMat := make([]float64, rows*cols)
	var wg sync.WaitGroup
	for y := 0; y < rows; y++ {
		wg.Add(1)
		go src.execPool(newMat, y, rows, cols, k, s, mode, &wg)
	}
	wg.Wait()
	emat.Dense = mat64.NewDense(rows, cols, newMat)
}

func (emat *ExMat) flatten(res []float64, y int, wg *sync.WaitGroup) {
	_, c := emat.Dims()
	for x := 0; x < c; x++ {
		res[y*c+x] = emat.At(y, x)
	}
	wg.Done()
}

// Flatten makes flat ExMat.
func (emat *ExMat) Flatten() *ExMat {
	r, c := emat.Dims()
	res := make([]float64, r*c)
	var wg sync.WaitGroup
	for y := 0; y < r; y++ {
		wg.Add(1)
		go emat.flatten(res, y, &wg)
	}
	wg.Wait()
	return NewExMat(1, r*c, res)
}

func (emat *ExMat) edgePad(rows, cols, newRows, newCols int, newMatrix []float64, w, y int, wg *sync.WaitGroup) {
	for x := 0; x < newCols; x++ {
		if y < w && x < w {
			newMatrix[y*newCols+x] = emat.At(0, 0)
		} else if y < w && x > w-1 && x < cols+w {
			newMatrix[y*newCols+x] = emat.At(0, x-w)
		} else if y < w && x > cols+w-1 {
			newMatrix[y*newCols+x] = emat.At(0, cols-1)
		} else if y > w-1 && y < rows+w && x < w {
			newMatrix[y*newCols+x] = emat.At(y-w, 0)
		} else if y > w-1 && y < rows+w && x > cols+w-1 {
			newMatrix[y*newCols+x] = emat.At(y-w, cols-1)
		} else if y > rows+w-1 && x < w {
			newMatrix[y*newCols+x] = emat.At(rows-1, 0)
		} else if y > rows+w-1 && x > w-1 && x < cols+w {
			newMatrix[y*newCols+x] = emat.At(rows-1, x-w)
		} else if y > rows+w-1 && x > cols+w-1 {
			newMatrix[y*newCols+x] = emat.At(rows-1, cols-1)
		} else {
			newMatrix[y*newCols+x] = emat.At(y-w, x-w)
		}
	}
	wg.Done()
}

// EdgePadding pads ExMat with edge's value.
func (emat *ExMat) EdgePadding(w int) *ExMat {
	r, c := emat.Dims()
	newRows := r + w*2
	newCols := c + w*2
	newMatrix := make([]float64, newRows*newCols)
	var wg sync.WaitGroup
	for y := 0; y < newRows; y++ {
		wg.Add(1)
		go emat.edgePad(r, c, newRows, newCols, newMatrix, w, y, &wg)
	}
	wg.Wait()
	return NewExMat(newRows, newCols, newMatrix)
}

func (emat *ExMat) zeroPad(rows, cols, newRows, newCols int, newMatrix []float64, w, y int, wg *sync.WaitGroup) {
	for x := 0; x < newCols; x++ {
		if y > w-1 && y < rows+w && x > w-1 && x < cols+w {
			newMatrix[y*newCols+x] = emat.At(y-w, x-w)
		} else {
			newMatrix[y*newCols+x] = 0.0
		}
	}
	wg.Done()
}

// ZeroPadding pads ExMat with zero.
func (emat *ExMat) ZeroPadding(w int) *ExMat {
	r, c := emat.Dims()
	newRows := r + w*2
	newCols := c + w*2
	newMatrix := make([]float64, newRows*newCols)
	var wg sync.WaitGroup
	for y := 0; y < newRows; y++ {
		wg.Add(1)
		go emat.zeroPad(r, c, newRows, newCols, newMatrix, w, y, &wg)
	}
	wg.Wait()
	return NewExMat(newRows, newCols, newMatrix)
}

func (emat *ExMat) makeCol(k, s, y, cols, sIdx int, res []float64, wg *sync.WaitGroup) {
	for x := 0; x < cols; x++ {
		part := emat.View(y*s, x*s, k, k).(*mat64.Dense)
		for i := 0; i < k; i++ {
			for j := 0; j < k; j++ {
				res[sIdx] = part.At(i, j)
				sIdx++
			}
		}
	}
	wg.Done()
}

// Im2Col make col matrix of ExMat.
func (emat *ExMat) Im2Col(k, s int) *ExMat {
	colSize := k * k
	r, c := emat.Dims()
	rows := (r-k)/s + 1
	cols := (c-k)/s + 1
	res := make([]float64, colSize*rows*cols)
	idx := 0
	var wg sync.WaitGroup
	for y := 0; y < rows; y++ {
		wg.Add(1)
		go emat.makeCol(k, s, y, cols, idx, res, &wg)
		idx += colSize * cols
	}
	wg.Wait()
	return NewExMat(rows*cols, colSize, res)
}

// Convolve2d convolves ExMat.
func (emat *ExMat) Convolve2d(s int, f *ExMat) (err error) {
	fr, fc := f.Dims()
	if fr != fc {
		return fmt.Errorf("Dimension mismatch %d, %d", fr, fc)
	}
	r, c := emat.Dims()
	rows := (r-fr)/s + 1
	cols := (c-fc)/s + 1
	in := emat.Im2Col(fr, s)
	flat := f.Flatten()
	var out mat64.Dense
	out.Mul(flat, in.T())
	exOut := NewExMatFromDense(&out)
	if err = emat.Reshape(rows, cols, exOut); err != nil {
		return
	}
	return nil
}
